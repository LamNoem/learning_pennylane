import pandas as pd
import numpy as onp

import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path


#### dataset ####
train_df = pd.read_csv("public_train.csv")
test_df = pd.read_csv("public_test.csv")

feature_cols = [f"x{i}" for i in range(1, 9)]

X = train_df[feature_cols].values
# [[-0.35297954 -0.66388116  0.88291667 ... -0.7295028   0.53392688
#    0.5160065 ]
#  [-0.35297954 -2.17705179  0.98667982 ... -0.6142338   0.55341669
#    0.05978894]
#  [-0.35297954  1.4905149   0.70137421 ... -0.10520634  1.40924075
#    1.4330569 ]
#  ...
#  [-0.35297954 -1.49565837  0.94320204 ... -1.2284721  -0.63864475
#    0.68236965]
#  [-0.35297954 -1.32807303  0.93029926 ... -0.79887272  0.52365099
#    0.38614567]
#  [-0.35297954 -1.69228655 -1.19470034 ... -0.3190028   0.06343364
#    0.34224323]]
y = train_df["label"].values
# print(y)
# [0 0 0 ... 0 0 0]

#data has 8 input features: x1, x2, x3, x4, x5, x6, x7, x8
# and one label : 0 or 1

#### Split training and validation data ####
#validattion uses portion of train data not test. only use test data for final split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)
# stratify keeps the class balance similar in the training and validation sets.

#### Create the quantum device ####
num_qubits = 8
num_layers = 5
entangler = "cz_brickwork_alternating" # "chain" or "brickwork" or "chain_alternating" or "brickwork_alternating" or "cz_brickwork_alternating"
measured_qubit = 0 # only one qubit can be measured, competition rule
reupload_data = True
seed=0
encoding_mode = "RY" 
rotation_order = "RY_RZ_RX"  # current
num_rotational_gates_per_qubit = 3 #RY and RZ, optional RX
use_initial_h = False
measurement_basis = "Z"  # Z was the original choice
# measurement_basis = "X"

n0_train = onp.sum(y_train == 0)
n1_train = onp.sum(y_train == 1)
total_train = len(y_train)

class_weight_0 = total_train / (2 * n0_train)
class_weight_1 = total_train / (2 * n1_train)

print("Class weight 0:", class_weight_0)
print("Class weight 1:", class_weight_1)

dev = qml.device("default.qubit", wires=num_qubits)
#We use 8 qubits because there are 8 features

### Encode the data directly### 


def encode_data(x):
    for i in range(num_qubits):
        if encoding_mode == "RY":
            qml.RY(x[i], wires=i)

        elif encoding_mode == "RY_RZ":
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)

        elif encoding_mode == "RX_RY_RZ":
            qml.RX(x[i], wires=i)
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)

        else:
            raise ValueError(f"Unknown encoding_mode: {encoding_mode}")

#This is direct encoding. No scaling. No normalization. No PCA. No get_angles.


# can be used before encoding, This starts every qubit in a superposition before data encoding.
def apply_initial_layer():
    if use_initial_h:
        for i in range(num_qubits):
            qml.Hadamard(wires=i)


def apply_measurement_basis():
    if measurement_basis == "Z":
        pass
    elif measurement_basis == "X":
        qml.Hadamard(wires=measured_qubit)
    else:
        raise ValueError(f"Unknown measurement_basis: {measurement_basis}")
### Define trainable layer ###

def apply_entanglement(layer_id):
    if entangler == "chain":
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    elif entangler == "brickwork":
        for i in range(0, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

        for i in range(1, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    elif entangler == "chain_alternating":
        if layer_id % 2 == 0:
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        else:
            for i in reversed(range(num_qubits - 1)):
                qml.CNOT(wires=[i + 1, i])

    elif entangler == "brickwork_alternating":
        if layer_id % 2 == 0:
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        else:
            # Odd layers: right-to-left brickwork
            for i in reversed(range(1, num_qubits - 1, 2)):
                qml.CNOT(wires=[i + 1, i])

            for i in reversed(range(0, num_qubits - 1, 2)):
                qml.CNOT(wires=[i + 1, i])
    elif entangler == "cz_brickwork_alternating":
        if layer_id % 2 == 0:
            for i in range(0, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])

            for i in range(1, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
        else:
            # Odd layers: right-to-left brickwork
            for i in reversed(range(1, num_qubits - 1, 2)):
                qml.CZ(wires=[i + 1, i])

            for i in reversed(range(0, num_qubits - 1, 2)):
                qml.CZ(wires=[i + 1, i])
    else:
        raise ValueError(f"Unknown entangler: {entangler}")

def trainable_layer(layer_weights, layer_id):
    # Trainable single-qubit gates
    for i in range(num_qubits):
        if rotation_order == "RY_RZ":
            qml.RY(layer_weights[i, 0], wires=i)
            qml.RZ(layer_weights[i, 1], wires=i)

        elif rotation_order == "RY_RZ_RX":
            qml.RY(layer_weights[i, 0], wires=i)
            qml.RZ(layer_weights[i, 1], wires=i)
            qml.RX(layer_weights[i, 2], wires=i)

        elif rotation_order == "RX_RY_RZ":
            qml.RX(layer_weights[i, 0], wires=i)
            qml.RY(layer_weights[i, 1], wires=i)
            qml.RZ(layer_weights[i, 2], wires=i)

        elif rotation_order == "RZ_RY_RX":
            qml.RZ(layer_weights[i, 0], wires=i)
            qml.RY(layer_weights[i, 1], wires=i)
            qml.RX(layer_weights[i, 2], wires=i)

        else:
            raise ValueError(f"Unknown rotation_order: {rotation_order}")
    

    # Entangling gates
    apply_entanglement(layer_id)

    #uses 7 two qubit gates per layer 
    # with 4 layers
    #  4 × 7 = 28 two-qubit gates, competition limit is 80

## full quantum circuit ## 

@qml.qnode(dev, interface="autograd")
def circuit(weights, bias, x):
    apply_initial_layer()
    ##encode_data(x), option to only encode data once.
    if not reupload_data:
        encode_data(x)

    for layer_id in range(num_layers):
        if reupload_data:
            encode_data(x) # each layer gets the freshly encoded data
        trainable_layer(weights[layer_id], layer_id)

    # In-circuit bias instead of classical + bias
    qml.RY(bias, wires=measured_qubit)

    #for qubit 0
    # PauliZ expectation near +1 → likely measured as 0
    #PauliZ expectation near -1 → likely measured as 1 why ?
    apply_measurement_basis()
    return qml.expval(qml.PauliZ(measured_qubit))


###Convert the measured value into class probability##
#PauliZ(0) returns a value between -1 and +1.
def prob_class_1(weights, bias, x):
    
    z = circuit(weights, bias, x)
    # -1 -> 1
    # 1 -> 0
    return (1 - z) / 2


#### Define balanced-accuracy-friendly loss #####
# binary cross entropy
def weighted_bce_loss(weights, bias, X_batch, y_batch):
    probs = [prob_class_1(weights, bias, x) for x in X_batch]
    probs = np.stack(probs)

    eps = 1e-7

    sample_weights = np.where(
        y_batch == 1,
        class_weight_1,
        class_weight_0
    )

    loss = -np.mean(
        sample_weights * (
            y_batch * np.log(probs + eps)
            + (1 - y_batch) * np.log(1 - probs + eps)
        )
    )

    return loss


##### Define prediction and validation functions ####
def predict(weights, bias, X_data):
    probs = [prob_class_1(weights, bias, x) for x in X_data]
    probs = onp.array([float(p) for p in probs])
    return (probs >= 0.5).astype(int)

def validation_balanced_accuracy(weights, bias):
    preds = predict(weights, bias, X_val)
    return balanced_accuracy_score(y_val, preds)



### Initialize the parameters ###
# num_layers × num_qubits × 2
# 4 × 8 × 2 = 64
# Plus one final bias rotation:
# 64 + 1 = 65 parameters

onp.random.seed(seed)

weights = 0.01 * np.array(
    onp.random.randn(num_layers, num_qubits, num_rotational_gates_per_qubit),
    requires_grad=True
)

bias = np.array(0.0, requires_grad=True)

###Train the model ###
lr = 0.02
opt = qml.AdamOptimizer(stepsize=lr) #try 0.01, 0.02, 0.03, 0.05

batch_size = 64 #64 #128 #32
num_epochs = 150

patience = 50
epochs_without_improvement = 0

best_bal_acc = 0
best_weights = None
best_bias = None

rng = onp.random.default_rng(seed)

for epoch in range(num_epochs):
    batch_idx = rng.choice(len(X_train), size=batch_size, replace=False)

    X_batch = X_train[batch_idx]
    y_batch = y_train[batch_idx]

    weights, bias = opt.step(
        lambda w, b: weighted_bce_loss(w, b, X_batch, y_batch),
        weights,
        bias
    )

    if (epoch + 1) % 5 == 0:
        val_bal_acc = validation_balanced_accuracy(weights, bias)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Validation balanced accuracy: {val_bal_acc:.4f}"
        )

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_weights = weights.copy()
            best_bias = bias.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 5

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break



##### TEST #####
if best_weights is None or best_bias is None:
    raise RuntimeError("Training finished without saving a best model.")
X_public = test_df[feature_cols].values
y_public = test_df["label"].values

public_preds = predict(best_weights, best_bias, X_public)
public_bal_acc = balanced_accuracy_score(y_public, public_preds)

print("Public test balanced accuracy:", public_bal_acc)




###### Convert trained parameters into weights.json #####
import json

flat_params = []

for layer_id in range(num_layers):
    for qubit_id in range(num_qubits):
        for gate_id in range(num_rotational_gates_per_qubit):
            flat_params.append(
                float(best_weights[layer_id, qubit_id, gate_id])
            )

flat_params.append(float(best_bias))

weights_json = {
    f"theta_{i}": value
    for i, value in enumerate(flat_params)
}

with open("weights.json", "w") as f:
    json.dump(weights_json, f, indent=2)

# This creates:

# theta_0
# theta_1
# theta_2
# ...
# theta_64

expected_num_params = num_layers * num_qubits * num_rotational_gates_per_qubit + 1

assert len(weights_json) == expected_num_params, (
    f"weights.json has {len(weights_json)} parameters, "
    f"but expected {expected_num_params}"
)

assert list(weights_json.keys()) == [
    f"theta_{i}" for i in range(expected_num_params)
], "theta keys are not sequential"

print("weights.json parameter count OK:", expected_num_params)

#### Translate the circuit into QASM #### 

#Encode x1...x8 directly as RY angles on 8 qubits → apply 4 layers of trainable RY/RZ gates plus CNOT entanglement → measure q[0] → train theta_0...theta_64 using training data only → submit QASM plus weights.json.
# Each row has eight raw features x1...x8. I encode them directly as rotation angles using RY(x_i) on qubit i, with no scaling, normalization, feature engineering, PCA, or data augmentation. The same direct encoding is repeated in each data re-uploading layer.



#### evaluation with 1024 shots ########
shot_dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(shot_dev, shots=1024)
def circuit_1024(weights, bias, x):
    if not reupload_data:
        encode_data(x)

    for layer_id in range(num_layers):
        if reupload_data:
            encode_data(x)

        trainable_layer(weights[layer_id], layer_id)

    qml.RY(bias, wires=measured_qubit)

    return qml.probs(wires=measured_qubit)

def predict_1024(weights, bias, X_data):
    probs_class_1 = []

    for x in X_data:
        probs = circuit_1024(weights, bias, x)
        probs_class_1.append(float(probs[1]))

    probs_class_1 = onp.array(probs_class_1)

    return (probs_class_1 >= 0.5).astype(int)


public_preds_1024 = predict_1024(best_weights, best_bias, X_public)
public_bal_acc_1024 = balanced_accuracy_score(y_public, public_preds_1024)

print("Public test balanced accuracy with 1024 shots:", public_bal_acc_1024)


# lr,bs,epochs,layers,entanglement,qubit measured,seed,val_bacc,test_bacc, 1024 bacc
results_path = Path("results_c1.csv")
tunings = [
    lr,
    batch_size,
    epoch + 1,
    num_layers,
    entangler,
    measured_qubit,
    seed,
    rotation_order,
    best_bal_acc,
    public_bal_acc,
    public_bal_acc_1024,
    encoding_mode,
    use_initial_h,
    measurement_basis
]
header = [
    "lr",
    "batch_size",
    "epochs_run",
    "num_layers",
    "entanglement",
    "measured_qubit",
    "seed",
    "rotation_order",
    "best_val_bal_acc",
    "public_bal_acc_exact",
    "public_bal_acc_1024",
    "encoding_mode",
    "initial_h",
    "measurement_basis"
]

write_header = not results_path.exists()
import csv
with open(results_path, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if write_header:
        writer.writerow(header)

    writer.writerow(tunings)

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit import qasm3


def build_qiskit_classifier(
    num_qubits=8,
    num_layers=4,
    measured_qubit=0,
    entangler="chain",      # "chain" or "brickwork"
    reupload_data=True      # True = encode x in every layer
):
    """
    Builds the same classifier architecture in Qiskit so we can export OpenQASM 3.0.

    Important:
    - This must match your PennyLane training circuit exactly.
    - If you train with chain entanglement, export with entangler="chain".
    - If you train with brickwork entanglement, export with entangler="brickwork".
    """

    if num_qubits > 8:
        raise ValueError("Competition rule violation: maximum number of qubits is 8.")

    if measured_qubit < 0 or measured_qubit >= num_qubits:
        raise ValueError("Measured qubit index is invalid.")

    if entangler not in ["chain", "brickwork", "chain_alternating", "brickwork_alternating"]:
        raise ValueError("entangler must be either 'chain' or 'brickwork' or 'chain_alternating' or 'brickwork_alternating'.")

    # One classical bit because the competition allows 1-qubit measurement.
    qc = QuantumCircuit(num_qubits, 1)

    # Data placeholders: x_0, x_1, ..., x_7
    # These represent the raw features x1...x8.
    x = [Parameter(f"x_{i}") for i in range(num_qubits)]

    # Trainable parameters:
    # Each layer has 8 qubits * 2 trainable rotations = 16 parameters.
    # Plus one final trainable bias rotation.
    num_theta = num_layers * num_qubits * num_rotational_gates_per_qubit + 1
    theta = [Parameter(f"theta_{i}") for i in range(num_theta)]

    theta_index = 0

    # If reupload_data=False, encode data once at the beginning.
    if not reupload_data:
        # todo: check if encoding_mode is RX_RY_RZ, RY_RZ, or RY, and apply the corresponding rotations.
        for q in range(num_qubits):
            qc.ry(x[q], q)

    for layer in range(num_layers):

        # Direct data encoding.
        # If reupload_data=True, we encode the same raw x values in every layer.
        if reupload_data:
            # todo: check if encoding_mode is RX_RY_RZ, RY_RZ, or RY, and apply the corresponding rotations.
            for q in range(num_qubits):
                qc.ry(x[q], q)

        # Trainable single-qubit rotations.
        # This matches your PennyLane code:
        # qml.RY(layer_weights[i, 0])
        # qml.RZ(layer_weights[i, 1])
        for q in range(num_qubits):
            qc.ry(theta[theta_index], q)
            theta_index += 1

            qc.rz(theta[theta_index], q)
            theta_index += 1

            if num_rotational_gates_per_qubit == 3:
                qc.rx(theta[theta_index], q)
                theta_index += 1


        # Entanglement pattern.
        if entangler == "chain":
            # Forward chain:
            # CX 0->1, 1->2, ..., 6->7
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)

        elif entangler == "brickwork":
            # First parallel group:
            # CX 0->1, 2->3, 4->5, 6->7
            for q in range(0, num_qubits - 1, 2):
                qc.cx(q, q + 1)

            # Second parallel group:
            # CX 1->2, 3->4, 5->6
            for q in range(1, num_qubits - 1, 2):
                qc.cx(q, q + 1)

        elif entangler == "chain_alternating":
            if layer % 2 == 0:
                # Even layers: forward chain
                # CX 0->1, 1->2, ..., 6->7
                for q in range(num_qubits - 1):
                    qc.cx(q, q + 1)
            else:
                # Odd layers: reverse chain
                # CX 7->6, 6->5, ..., 1->0
                for q in reversed(range(num_qubits - 1)):
                    qc.cx(q + 1, q)

        elif entangler == "brickwork_alternating":
            if layer % 2 == 0:
                # Even layers: forward brickwork

                # First parallel group:
                # CX 0->1, 2->3, 4->5, 6->7
                for q in range(0, num_qubits - 1, 2):
                    qc.cx(q, q + 1)

                # Second parallel group:
                # CX 1->2, 3->4, 5->6
                for q in range(1, num_qubits - 1, 2):
                    qc.cx(q, q + 1)

            else:
                # Odd layers: reverse brickwork

                # First reverse group:
                # CX 6->5, 4->3, 2->1
                for q in reversed(range(1, num_qubits - 1, 2)):
                    qc.cx(q + 1, q)

                # Second reverse group:
                # CX 7->6, 5->4, 3->2, 1->0
                for q in reversed(range(0, num_qubits - 1, 2)):
                    qc.cx(q + 1, q)

        else:
            raise ValueError(f"Unknown entangler: {entangler}")

    # Final trainable bias rotation on the measured qubit.
    qc.ry(theta[theta_index], measured_qubit)

    # Measure exactly one qubit into exactly one classical bit.
    qc.measure(measured_qubit, 0)

    return qc, theta, x

def check_competition_constraints(qc, max_depth=50, max_two_qubit_gates=80):
    """
    Checks the competition constraints:
    - <= 8 qubits
    - depth <= 50
    - two-qubit gates <= 80
    - allowed gates only
    - one measured qubit
    """

    allowed_ops = {
        "x", "y", "z",
        "h", "s", "t",
        "rx", "ry", "rz",
        "cx", "cz",
        "measure"
    }

    ops = qc.count_ops()
    used_ops = set(ops.keys())

    bad_ops = used_ops - allowed_ops
    if bad_ops:
        raise ValueError(f"Competition rule violation: forbidden gates found: {bad_ops}")

    if qc.num_qubits > 8:
        raise ValueError(f"Competition rule violation: {qc.num_qubits} qubits used, max is 8.")

    two_qubit_count = 0
    measurement_count = 0

    for instruction in qc.data:
        op_name = instruction.operation.name
        num_qargs = len(instruction.qubits)

        if op_name in {"cx", "cz"}:
            two_qubit_count += 1

        if op_name == "measure":
            measurement_count += 1

        if num_qargs == 2 and op_name not in {"cx", "cz"}:
            raise ValueError(f"Competition rule violation: unsupported two-qubit gate {op_name}")

    if two_qubit_count > max_two_qubit_gates:
        raise ValueError(
            f"Competition rule violation: {two_qubit_count} two-qubit gates used, "
            f"max is {max_two_qubit_gates}."
        )

    if measurement_count != 1:
        raise ValueError(
            f"Competition rule violation: {measurement_count} measurements found. "
            "Only one measured qubit is allowed."
        )

    depth = qc.depth()

    if depth > max_depth:
        raise ValueError(
            f"Competition rule violation: circuit depth is {depth}, max is {max_depth}."
        )

    print("Circuit passes basic competition checks.")
    print("Qubits:", qc.num_qubits)
    print("Depth:", depth)
    print("Two-qubit gates:", two_qubit_count)
    print("Gate counts:", dict(ops))

def save_classifier_qasm(
    filename="classifier.qasm",
    num_qubits=8,
    num_layers=4,
    measured_qubit=0,
    entangler="chain",
    reupload_data=True
):
    qc, theta, x = build_qiskit_classifier(
        num_qubits=num_qubits,
        num_layers=num_layers,
        measured_qubit=measured_qubit,
        entangler=entangler,
        reupload_data=reupload_data
    )

    check_competition_constraints(qc)

    qasm_text = qasm3.dumps(qc)

    with open(filename, "w") as f:
        f.write(qasm_text)

    print(f"Saved {filename}")
    print(f"Measured qubit index: {measured_qubit}")
    print(f"Number of theta parameters: {len(theta)}")
    print(f"Expected weights.json keys: theta_0 through theta_{len(theta) - 1}")

    return qc, qasm_text





qc, qasm_text = save_classifier_qasm(
    filename="classifier.qasm",
    num_qubits=num_qubits,
    num_layers=num_layers,
    measured_qubit=measured_qubit,
    entangler=entangler,
    reupload_data=reupload_data
)

# print(qc.draw(output="text"))