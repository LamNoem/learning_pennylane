"""
debug_quantum_model.py

Debug saved quantum-classifier weights for the competition.

This script loads:
- public_train.csv
- public_test.csv
- weights.json

It reconstructs the same PennyLane circuit and compares:
1. Exact PauliZ expectation-based P(class 1)
2. Exact qml.probs-based P(class 1)
3. PennyLane finite-shot qml.probs-based P(class 1)
4. Manual 1024-shot binomial simulation from exact probabilities

Use this to diagnose why exact balanced accuracy may look good while
1024-shot balanced accuracy looks wrong.

Example for your current model:

python debug_quantum_model.py ^
    --train_csv public_train.csv ^
    --test_csv public_test.csv ^
    --weights_json weights.json ^
    --num_layers 4 ^
    --entangler chain ^
    --measured_qubit 0 ^
    --shots 1024 ^
    --debug_rows 20

On macOS/Linux, replace ^ with backslash line continuations, or run it on one line.

python debug_quantum_model.py --num_layers 4 --entangler chain --measured_qubit 0 --shots 1024 --debug_rows 20

"""

import argparse
import json
from pathlib import Path

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="public_train.csv")
    parser.add_argument("--test_csv", type=str, default="public_test.csv")
    parser.add_argument("--weights_json", type=str, default="weights.json")

    parser.add_argument("--num_qubits", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--measured_qubit", type=int, default=0)

    parser.add_argument(
        "--entangler",
        type=str,
        default="chain",
        choices=["chain", "brickwork"],
    )

    parser.add_argument(
        "--encode_once",
        action="store_true",
        help="Use this only if your trained model encoded the data once before all layers. Default is data re-uploading.",
    )

    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--split",
        type=str,
        default="public",
        choices=["public", "train"],
        help="Which dataset to debug: public uses public_test.csv, train uses public_train.csv.",
    )

    parser.add_argument(
        "--debug_rows",
        type=int,
        default=20,
        help="Number of rows to print detailed probability comparisons for.",
    )

    parser.add_argument(
        "--max_eval_rows",
        type=int,
        default=0,
        help="If > 0, only evaluate this many rows for speed.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="shot_debug_rows.csv",
        help="CSV file for row-level debug output.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Load weights
# -----------------------------------------------------------------------------

def load_weights(weights_path, num_layers, num_qubits):
    weights_path = Path(weights_path)

    with open(weights_path, "r") as f:
        weight_dict = json.load(f)

    expected_num_params = num_layers * num_qubits * 2 + 1
    expected_keys = [f"theta_{i}" for i in range(expected_num_params)]

    missing = [k for k in expected_keys if k not in weight_dict]
    extra = [k for k in weight_dict.keys() if k not in expected_keys]

    if missing:
        raise ValueError(f"Missing theta keys in weights.json: {missing[:10]}...")

    if extra:
        print(f"Warning: extra theta keys found and ignored: {extra[:10]}...")

    flat = onp.array([float(weight_dict[k]) for k in expected_keys], dtype=float)

    layer_params = flat[:-1].reshape(num_layers, num_qubits, 2)
    bias = flat[-1]

    weights = np.array(layer_params, requires_grad=False)
    bias = np.array(bias, requires_grad=False)

    print("Loaded weights:", weights_path)
    print("Expected parameters:", expected_num_params)
    print("Loaded layer weights shape:", weights.shape)
    print("Loaded bias:", float(bias))

    return weights, bias


# -----------------------------------------------------------------------------
# Build PennyLane circuits
# -----------------------------------------------------------------------------

def make_circuit_fns(num_qubits, num_layers, measured_qubit, entangler, reupload_data, shots):
    dev_exact = qml.device("default.qubit", wires=num_qubits)

    # This may print a deprecation warning in newer PennyLane versions, but it is
    # still useful because it matches the style of your current 1024-shot check.
    dev_shots = qml.device("default.qubit", wires=num_qubits, shots=shots)

    def encode_data(x):
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)

    def apply_entanglement():
        if entangler == "chain":
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        elif entangler == "brickwork":
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

        else:
            raise ValueError(f"Unknown entangler: {entangler}")

    def trainable_layer(layer_weights):
        for i in range(num_qubits):
            qml.RY(layer_weights[i, 0], wires=i)
            qml.RZ(layer_weights[i, 1], wires=i)

        apply_entanglement()

    def apply_full_circuit(weights, bias, x):
        if not reupload_data:
            encode_data(x)

        for layer_id in range(num_layers):
            if reupload_data:
                encode_data(x)

            trainable_layer(weights[layer_id])

        qml.RY(bias, wires=measured_qubit)

    @qml.qnode(dev_exact)
    def circuit_expval(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        return qml.expval(qml.PauliZ(measured_qubit))

    @qml.qnode(dev_exact)
    def circuit_probs_exact(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        return qml.probs(wires=measured_qubit)

    @qml.qnode(dev_shots)
    def circuit_probs_shots(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        return qml.probs(wires=measured_qubit)

    return circuit_expval, circuit_probs_exact, circuit_probs_shots


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def p1_from_expval(z):
    return (1.0 - float(z)) / 2.0


def evaluate_exact_expval(circuit_expval, weights, bias, X):
    probs = []

    for x in X:
        z = circuit_expval(weights, bias, x)
        probs.append(p1_from_expval(z))

    probs = onp.array(probs)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def evaluate_probs(circuit_probs, weights, bias, X):
    probs_1 = []

    for x in X:
        probs = circuit_probs(weights, bias, x)
        probs_1.append(float(probs[1]))

    probs_1 = onp.array(probs_1)
    preds = (probs_1 >= 0.5).astype(int)
    return probs_1, preds


def evaluate_manual_shots(exact_probs_1, shots=1024, seed=0):
    rng = onp.random.default_rng(seed)
    counts_1 = rng.binomial(shots, exact_probs_1)
    probs_1 = counts_1 / shots
    preds = (counts_1 >= shots / 2).astype(int)
    return probs_1, preds, counts_1


def print_metrics(name, y_true, preds):
    bacc = balanced_accuracy_score(y_true, preds)
    recalls = recall_score(y_true, preds, average=None, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Balanced accuracy: {bacc:.6f}")
    print(f"Recall class 0:    {recalls[0]:.6f}")
    print(f"Recall class 1:    {recalls[1]:.6f}")
    print("Confusion matrix, rows=true, cols=pred:")
    print(cm)

    return bacc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    reupload_data = not args.encode_once

    if args.measured_qubit < 0 or args.measured_qubit >= args.num_qubits:
        raise ValueError("measured_qubit must be between 0 and num_qubits - 1")

    print("\nConfiguration")
    print("-------------")
    print("num_qubits:      ", args.num_qubits)
    print("num_layers:      ", args.num_layers)
    print("measured_qubit:  ", args.measured_qubit)
    print("entangler:       ", args.entangler)
    print("reupload_data:   ", reupload_data)
    print("shots:           ", args.shots)
    print("seed:            ", args.seed)

    feature_cols = [f"x{i}" for i in range(1, args.num_qubits + 1)]

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if args.split == "public":
        df = test_df
        print("\nUsing public test set:", args.test_csv)
    else:
        df = train_df
        print("\nUsing train set:", args.train_csv)

    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    if args.max_eval_rows and args.max_eval_rows > 0:
        X = X[: args.max_eval_rows]
        y = y[: args.max_eval_rows]
        print(f"Evaluating only first {len(X)} rows.")

    weights, bias = load_weights(args.weights_json, args.num_layers, args.num_qubits)

    circuit_expval, circuit_probs_exact, circuit_probs_shots = make_circuit_fns(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        measured_qubit=args.measured_qubit,
        entangler=args.entangler,
        reupload_data=reupload_data,
        shots=args.shots,
    )

    print("\nEvaluating exact expval model...")
    p_expval, pred_expval = evaluate_exact_expval(circuit_expval, weights, bias, X)

    print("Evaluating exact qml.probs model...")
    p_probs_exact, pred_probs_exact = evaluate_probs(circuit_probs_exact, weights, bias, X)

    print("Evaluating PennyLane finite-shot qml.probs model...")
    p_probs_shots, pred_probs_shots = evaluate_probs(circuit_probs_shots, weights, bias, X)

    print("Evaluating manual binomial finite-shot model...")
    p_manual_shots, pred_manual_shots, counts_manual = evaluate_manual_shots(
        p_expval,
        shots=args.shots,
        seed=args.seed,
    )

    bacc_expval = print_metrics("Exact expval -> P(1)=(1-z)/2", y, pred_expval)
    bacc_probs_exact = print_metrics("Exact qml.probs", y, pred_probs_exact)
    bacc_probs_shots = print_metrics(f"PennyLane qml.probs with {args.shots} shots", y, pred_probs_shots)
    bacc_manual = print_metrics(f"Manual binomial simulation with {args.shots} shots", y, pred_manual_shots)

    print("\nAgreement checks")
    print("----------------")
    print("Max |p_expval - p_probs_exact|:", float(onp.max(onp.abs(p_expval - p_probs_exact))))
    print("Mean |p_expval - p_probs_exact|:", float(onp.mean(onp.abs(p_expval - p_probs_exact))))
    print("Exact predictions agree with exact-probs predictions:", float(onp.mean(pred_expval == pred_probs_exact)))
    print("Exact predictions agree with PennyLane-shot predictions:", float(onp.mean(pred_expval == pred_probs_shots)))
    print("Exact predictions agree with manual-shot predictions:", float(onp.mean(pred_expval == pred_manual_shots)))

    print("\nFirst rows debug")
    print("----------------")
    n_debug = min(args.debug_rows, len(X))

    rows = []

    for i in range(n_debug):
        row = {
            "row": i,
            "label": int(y[i]),
            "p_expval": float(p_expval[i]),
            "pred_expval": int(pred_expval[i]),
            "p_probs_exact": float(p_probs_exact[i]),
            "pred_probs_exact": int(pred_probs_exact[i]),
            "p_probs_shots": float(p_probs_shots[i]),
            "pred_probs_shots": int(pred_probs_shots[i]),
            "p_manual_shots": float(p_manual_shots[i]),
            "pred_manual_shots": int(pred_manual_shots[i]),
            "manual_count_1": int(counts_manual[i]),
        }
        rows.append(row)

        print(
            f"row {i:4d} | "
            f"y={row['label']} | "
            f"p_expval={row['p_expval']:.4f}, pred={row['pred_expval']} | "
            f"p_probs_exact={row['p_probs_exact']:.4f}, pred={row['pred_probs_exact']} | "
            f"p_PL_{args.shots}={row['p_probs_shots']:.4f}, pred={row['pred_probs_shots']} | "
            f"p_manual_{args.shots}={row['p_manual_shots']:.4f}, pred={row['pred_manual_shots']}"
        )

    debug_df = pd.DataFrame(rows)
    debug_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved row-level debug CSV to: {args.output_csv}")

    print("\nInterpretation guide")
    print("--------------------")
    print("1. If exact expval and exact qml.probs disagree, your P(1) mapping is wrong.")
    print("2. If exact qml.probs and PennyLane-shot probs disagree badly, the shot QNode is the issue.")
    print("3. If exact expval and manual shots agree, the model itself is probably fine.")
    print("4. If all shot methods are much worse, many probabilities may be too close to 0.5.")

    print("\nSummary")
    print("-------")
    print(f"Exact expval bacc:        {bacc_expval:.6f}")
    print(f"Exact qml.probs bacc:     {bacc_probs_exact:.6f}")
    print(f"PennyLane shot bacc:      {bacc_probs_shots:.6f}")
    print(f"Manual shot bacc:         {bacc_manual:.6f}")


if __name__ == "__main__":
    main()
