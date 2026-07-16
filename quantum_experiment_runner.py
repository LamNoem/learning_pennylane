"""
quantum_experiment_runner.py

Run a prioritized set of quantum-classifier experiments and save results as each
combination finishes.

This is a cleaned-up, resumable version of your single-run script. It keeps the
same core idea:

- load public_train.csv and public_test.csv
- split public_train.csv into train/validation
- encode x1...x8 directly into the circuit
- train theta_0, theta_1, ... only on the training split
- choose best weights by validation balanced accuracy
- evaluate public exact and public 1024-shot behavior
- generate matching classifier.qasm and weights.json files
- check competition rules before training each combination
- append one row to the results CSV after every combination

Example usage
-------------

Run from the beginning:
    python quantum_experiment_runner.py

Only show the experiment plan without training:
    python quantum_experiment_runner.py --dry-run

Start from combination 20:
    python quantum_experiment_runner.py --start-index 20

Run only 5 combinations starting at index 20:
    python quantum_experiment_runner.py --start-index 20 --max-runs 5

Skip combinations already present in the results CSV:
    python quantum_experiment_runner.py --skip-completed

Use PennyLane finite-shot evaluation instead of manual binomial shot simulation:
    python quantum_experiment_runner.py --shot-mode pennylane

Important note
--------------
The QASM generator here is deliberately explicit. It matches:
- encoding_mode
- use_initial_h
- rotation_order
- entangler, including CZ variants
- final bias rotation
- measurement_basis
- measured_qubit

This avoids the common mistake where the PennyLane training circuit and the QASM
submission circuit silently become different.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# Qiskit is used only for QASM generation and resource checks.
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NUM_QUBITS = 8
FEATURE_COLS = [f"x{i}" for i in range(1, NUM_QUBITS + 1)]

ALLOWED_OPS = {
    "x", "y", "z",
    "h", "s", "t",
    "rx", "ry", "rz",
    "cx", "cz",
    "measure",
}

# Trainable rotation orders. The order here determines the theta order.
ROTATION_ORDERS: Dict[str, Tuple[str, ...]] = {
    "RY_RZ": ("ry", "rz"),
    "RY_RZ_RX": ("ry", "rz", "rx"),
    "RX_RY_RZ": ("rx", "ry", "rz"),
    "RZ_RY_RX": ("rz", "ry", "rx"),
    "RY_RX_RZ": ("ry", "rx", "rz"),
}

# Direct data-encoding modes. These use the raw x_i values directly as angles.
ENCODING_MODES: Dict[str, Tuple[str, ...]] = {
    "RY": ("ry",),
    "RY_RZ": ("ry", "rz"),
    "RX_RY_RZ": ("rx", "ry", "rz"),
}

ENTANGLERS = {
    "chain",
    "brickwork",
    "chain_alternating",
    "brickwork_alternating",
    "cz_chain",
    "cz_brickwork",
    "cz_chain_alternating",
    "cz_brickwork_alternating",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment combination."""

    lr: float
    batch_size: int
    num_epochs: int
    patience: int
    num_layers: int
    entangler: str
    measured_qubit: int
    seed: int
    rotation_order: str
    encoding_mode: str
    use_initial_h: bool
    measurement_basis: str
    reupload_data: bool = True
    init_scale: float = 0.01
    class_weight_1_multiplier: float = 1.0

    @property
    def num_train_rot_per_qubit(self) -> int:
        return len(ROTATION_ORDERS[self.rotation_order])

    @property
    def num_trainable_params(self) -> int:
        # num_layers × num_qubits × rotations-per-qubit + final bias rotation
        return self.num_layers * NUM_QUBITS * self.num_train_rot_per_qubit + 1

    def key(self) -> str:
        """Stable key used for skip-completed/resume behavior."""
        parts = [
            f"lr={self.lr}",
            f"bs={self.batch_size}",
            f"epochs={self.num_epochs}",
            f"patience={self.patience}",
            f"layers={self.num_layers}",
            f"ent={self.entangler}",
            f"mq={self.measured_qubit}",
            f"seed={self.seed}",
            f"rot={self.rotation_order}",
            f"enc={self.encoding_mode}",
            f"h={self.use_initial_h}",
            f"mb={self.measurement_basis}",
            f"reupload={self.reupload_data}",
            f"init={self.init_scale}",
            f"cw1m={self.class_weight_1_multiplier}",
        ]
        return "|".join(parts)


# -----------------------------------------------------------------------------
# Experiment plan
# -----------------------------------------------------------------------------

def dedupe_configs(configs: Iterable[ExperimentConfig]) -> List[ExperimentConfig]:
    """Remove duplicate configs while preserving the first occurrence/order."""
    seen = set()
    out: List[ExperimentConfig] = []
    for cfg in configs:
        k = cfg.key()
        if k not in seen:
            seen.add(k)
            out.append(cfg)
    return out

##### not currently used ##########################
def build_custom_experiment_plan(**kwargs) -> ExperimentConfig:
    """
    Build a custom experiment configuration based on the best known parameters.
    This configuration is designed to be a strong candidate for the quantum classifier.
    """
    # Current best baseline from your CSV:
    # 5 layers, brickwork_alternating, 3 trainable rotations, RY encoding.
    def base(**kwargs) -> ExperimentConfig:
        values = dict(
            lr=0.02,
            batch_size=64,
            num_epochs=200,
            patience=50,
            num_layers=5,
            entangler="brickwork_alternating",
            measured_qubit=0,
            seed=0,
            rotation_order="RY_RZ_RX",
            encoding_mode="RY",
            use_initial_h=False,
            measurement_basis="Z",
            reupload_data=True,
            init_scale=0.01,
            class_weight_1_multiplier=1.0,
        )
        values.update(kwargs)
        return ExperimentConfig(**values)

    return base(**kwargs)

def build_experiment_plan() -> List[ExperimentConfig]:
    """
    Build combinations from most promising to least promising.

    This is not a huge blind grid. It is prioritized from your current results:
    - best family so far: 5-layer brickwork_alternating, RY_RZ_RX, RY encoding
    - then measured-qubit search
    - then seed search
    - then LR/depth variants
    - then encoding, measurement-basis, initial-H, rotation-order, and CZ tests
    """
    configs: List[ExperimentConfig] = []

    # Current best baseline from your CSV:
    # 5 layers, brickwork_alternating, 3 trainable rotations, RY encoding.
    def base(**kwargs) -> ExperimentConfig:
        values = dict(
            lr=0.02,
            batch_size=64,
            num_epochs=200,
            patience=50,
            num_layers=5,
            entangler="brickwork_alternating",
            measured_qubit=0,
            seed=0,
            rotation_order="RY_RZ_RX",
            encoding_mode="RY",
            use_initial_h=False,
            measurement_basis="Z",
            reupload_data=True,
            init_scale=0.01,
            class_weight_1_multiplier=1.0,
        )
        values.update(kwargs)
        return ExperimentConfig(**values)

    ##############changing one thing at a time from best arch so far (after combine the best changes) ##################
    # 1) Measured-qubit search on the best architecture.
    # q0 and q1 already looked best, so they appear first.
    for mq in [0, 1, 3, 4, 2, 5, 6, 7]:
        configs.append(base(measured_qubit=mq))

    # 2) Seed search for the best measured qubits seen so far.
    for mq in [0]:
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            configs.append(base(measured_qubit=mq, seed=seed))

    # 3) Learning-rate tuning around the best architecture.
    for lr in [0.02, 0.015, 0.01, 0.025, 0.03, 0.005]:
        for mq in [0]:
            configs.append(base(lr=lr, measured_qubit=mq))

    # 5) Encoding variants. These add depth but no trainable parameters.
    for encoding in ["RY_RZ", "RX_RY_RZ"]:
        for lr in [0.015, 0.01, 0.02]:
            for mq in [0]:
                configs.append(base(encoding_mode=encoding, lr=lr, measured_qubit=mq, num_epochs=200, patience=60))

    # 8) Rotation-order variants. Same number of parameters, different noncommuting order.
    for rot in ["RX_RY_RZ", "RZ_RY_RX", "RY_RX_RZ"]:
        for mq in [0]:
            configs.append(base(rotation_order=rot, measured_qubit=mq))

    # 6) Measurement basis. X basis adds H right before the one-qubit measurement.
    for mq in [0]:
        configs.append(base(measurement_basis="X", measured_qubit=mq))

    # 7) Initial H layer. This starts each qubit in superposition before encoding.
    for mb in ["Z", "X"]:
        for mq in [0]:
            configs.append(base(use_initial_h=True, measurement_basis=mb, measured_qubit=mq))


    # 9) CZ-based entanglement variants. CZ is allowed and sometimes behaves differently.
    for ent in ["cz_brickwork_alternating", "cz_brickwork", "cz_chain_alternating", "cz_chain"]:
        for mq in [0]:
            configs.append(base(entangler=ent, measured_qubit=mq))

    
    # 11) A few class-weight nudges for the best family.
    # If minority-class recall is weak, these can help balanced accuracy.
    for mult in [1.1, 1.2, 0.9]:
        for mq in [0]:
            configs.append(base(class_weight_1_multiplier=mult, measured_qubit=mq))

    ############# end of changing one at a time ######################

    #######now covering bases ########################

    # 4) Slightly deeper brickwork_alternating models.
    # These may or may not pass the depth rule; the script checks before training.
    for lr, batch in [(0.01, 64), (0.015, 64), (0.01, 128), (0.005, 64)]:
        for mq in [0]:
            configs.append(base(num_layers=6, lr=lr, batch_size=batch, measured_qubit=mq, num_epochs=250, patience=75))

    # 10) Two-rotation simpler variants. They may generalize better even if less expressive.
    for ent in ["brickwork_alternating", "brickwork", "chain", "chain_alternating"]:
        for layers in [5, 6]:
            for mq in [0]:
                configs.append(base(num_layers=layers, entangler=ent, rotation_order="RY_RZ", lr=0.02, measured_qubit=mq))


    ######### you can comment out above and plan custum experiments below ####################
    

    return dedupe_configs(configs)


# -----------------------------------------------------------------------------
# QASM/Qiskit circuit generation
# -----------------------------------------------------------------------------

def qiskit_apply_encoding(qc: QuantumCircuit, x: List[Parameter], cfg: ExperimentConfig) -> None:
    """Apply direct data encoding. This must match PennyLane encode_data."""
    gates = ENCODING_MODES[cfg.encoding_mode]
    for q in range(NUM_QUBITS):
        for gate in gates:
            if gate == "rx":
                qc.rx(x[q], q)
            elif gate == "ry":
                qc.ry(x[q], q)
            elif gate == "rz":
                qc.rz(x[q], q)
            else:
                raise ValueError(f"Unsupported encoding gate: {gate}")


def qiskit_apply_trainable_rotations(
    qc: QuantumCircuit,
    theta: List[Parameter],
    theta_index: int,
    q: int,
    cfg: ExperimentConfig,
) -> int:
    """Apply trainable rotations in exactly the same theta order as PennyLane."""
    gates = ROTATION_ORDERS[cfg.rotation_order]
    for gate in gates:
        if gate == "rx":
            qc.rx(theta[theta_index], q)
        elif gate == "ry":
            qc.ry(theta[theta_index], q)
        elif gate == "rz":
            qc.rz(theta[theta_index], q)
        else:
            raise ValueError(f"Unsupported trainable rotation gate: {gate}")
        theta_index += 1
    return theta_index


def qiskit_apply_entanglement(qc: QuantumCircuit, layer: int, cfg: ExperimentConfig) -> None:
    """Apply entanglement pattern. This must match PennyLane apply_entanglement."""
    ent = cfg.entangler

    if ent == "chain":
        for q in range(NUM_QUBITS - 1):
            qc.cx(q, q + 1)

    elif ent == "brickwork":
        for q in range(0, NUM_QUBITS - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, NUM_QUBITS - 1, 2):
            qc.cx(q, q + 1)

    elif ent == "chain_alternating":
        if layer % 2 == 0:
            for q in range(NUM_QUBITS - 1):
                qc.cx(q, q + 1)
        else:
            for q in reversed(range(NUM_QUBITS - 1)):
                qc.cx(q + 1, q)

    elif ent == "brickwork_alternating":
        if layer % 2 == 0:
            for q in range(0, NUM_QUBITS - 1, 2):
                qc.cx(q, q + 1)
            for q in range(1, NUM_QUBITS - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in reversed(range(1, NUM_QUBITS - 1, 2)):
                qc.cx(q + 1, q)
            for q in reversed(range(0, NUM_QUBITS - 1, 2)):
                qc.cx(q + 1, q)

    elif ent == "cz_chain":
        for q in range(NUM_QUBITS - 1):
            qc.cz(q, q + 1)

    elif ent == "cz_brickwork":
        for q in range(0, NUM_QUBITS - 1, 2):
            qc.cz(q, q + 1)
        for q in range(1, NUM_QUBITS - 1, 2):
            qc.cz(q, q + 1)

    elif ent == "cz_chain_alternating":
        # CZ is symmetric, but we keep layer-dependent order to match the idea.
        if layer % 2 == 0:
            for q in range(NUM_QUBITS - 1):
                qc.cz(q, q + 1)
        else:
            for q in reversed(range(NUM_QUBITS - 1)):
                qc.cz(q + 1, q)

    elif ent == "cz_brickwork_alternating":
        # CZ is symmetric, but we keep the same alternating pattern structure.
        if layer % 2 == 0:
            for q in range(0, NUM_QUBITS - 1, 2):
                qc.cz(q, q + 1)
            for q in range(1, NUM_QUBITS - 1, 2):
                qc.cz(q, q + 1)
        else:
            for q in reversed(range(1, NUM_QUBITS - 1, 2)):
                qc.cz(q + 1, q)
            for q in reversed(range(0, NUM_QUBITS - 1, 2)):
                qc.cz(q + 1, q)

    else:
        raise ValueError(f"Unknown entangler: {ent}")


def build_qiskit_classifier(cfg: ExperimentConfig) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """
    Build the same classifier architecture in Qiskit so we can export OpenQASM 3.0.

    Important:
    - This must match your PennyLane training circuit exactly.
    - The QASM contains symbolic data placeholders x_0...x_7.
    - The QASM contains symbolic trainable placeholders theta_0, theta_1, ...
    """
    if NUM_QUBITS > 8:
        raise ValueError("Competition rule violation: maximum number of qubits is 8.")
    if cfg.measured_qubit < 0 or cfg.measured_qubit >= NUM_QUBITS:
        raise ValueError("Measured qubit index is invalid.")
    if cfg.entangler not in ENTANGLERS:
        raise ValueError(f"Unknown entangler: {cfg.entangler}")
    if cfg.encoding_mode not in ENCODING_MODES:
        raise ValueError(f"Unknown encoding_mode: {cfg.encoding_mode}")
    if cfg.rotation_order not in ROTATION_ORDERS:
        raise ValueError(f"Unknown rotation_order: {cfg.rotation_order}")
    if cfg.measurement_basis not in {"Z", "X"}:
        raise ValueError(f"Unknown measurement_basis: {cfg.measurement_basis}")

    # One classical bit because the competition allows 1-qubit measurement.
    qc = QuantumCircuit(NUM_QUBITS, 1)

    # Data placeholders: x_0, x_1, ..., x_7
    # These represent the raw features x1...x8.
    x = [Parameter(f"x_{i}") for i in range(NUM_QUBITS)]

    # Trainable parameters: theta_0, theta_1, ...
    theta = [Parameter(f"theta_{i}") for i in range(cfg.num_trainable_params)]
    theta_index = 0

    # Optional initial H layer: starts every qubit in superposition before encoding.
    if cfg.use_initial_h:
        for q in range(NUM_QUBITS):
            qc.h(q)

    # If reupload_data=False, encode data once at the beginning.
    if not cfg.reupload_data:
        qiskit_apply_encoding(qc, x, cfg)

    for layer in range(cfg.num_layers):
        # Direct data encoding.
        # If reupload_data=True, we encode the same raw x values in every layer.
        if cfg.reupload_data:
            qiskit_apply_encoding(qc, x, cfg)

        # Trainable single-qubit rotations.
        for q in range(NUM_QUBITS):
            theta_index = qiskit_apply_trainable_rotations(qc, theta, theta_index, q, cfg)

        # Entangling gates.
        qiskit_apply_entanglement(qc, layer, cfg)

    # Final trainable bias rotation on the measured qubit.
    qc.ry(theta[theta_index], cfg.measured_qubit)
    theta_index += 1

    if theta_index != len(theta):
        raise RuntimeError(f"Theta mismatch: used {theta_index}, expected {len(theta)}")

    # Measurement-basis choice.
    # Z: normal measurement. X: apply H before computational-basis measurement.
    if cfg.measurement_basis == "X":
        qc.h(cfg.measured_qubit)

    # Measure exactly one qubit into exactly one classical bit.
    qc.measure(cfg.measured_qubit, 0)

    return qc, theta, x


def qasm3_with_angle_inputs(qc: QuantumCircuit) -> str:
    """
    Export QASM 3.0 and post-process x_i/theta_i declarations to angle[64].

    Qiskit may export symbolic gate parameters as float[64]. Since these values are
    used directly as rotation angles, angle[64] is clearer for this competition.
    """
    qasm_text = qasm3.dumps(qc)
    qasm_text = re.sub(r"input float\[64\] (theta_\d+);", r"input angle[64] \1;", qasm_text)
    qasm_text = re.sub(r"input float\[64\] (x_\d+);", r"input angle[64] \1;", qasm_text)
    return qasm_text


def check_competition_constraints(qc: QuantumCircuit) -> Dict[str, object]:
    """
    Checks the competition constraints:
    - <= 8 qubits
    - depth <= 50
    - two-qubit gates <= 80
    - allowed gates only
    - one measured qubit
    """
    ops = qc.count_ops()
    used_ops = set(ops.keys())
    bad_ops = sorted(used_ops - ALLOWED_OPS)

    two_qubit_count = 0
    measurement_count = 0
    unsupported_two_qubit = []

    for instruction in qc.data:
        op_name = instruction.operation.name
        num_qargs = len(instruction.qubits)

        if op_name in {"cx", "cz"}:
            two_qubit_count += 1
        if op_name == "measure":
            measurement_count += 1
        if num_qargs == 2 and op_name not in {"cx", "cz"}:
            unsupported_two_qubit.append(op_name)

    depth = qc.depth()
    violations = []

    if qc.num_qubits > 8:
        violations.append(f"qubits>{8}")
    if depth > 50:
        violations.append(f"depth>{50}")
    if two_qubit_count > 80:
        violations.append(f"two_qubit>{80}")
    if bad_ops:
        violations.append("bad_ops:" + "+".join(bad_ops))
    if unsupported_two_qubit:
        violations.append("unsupported_two_qubit:" + "+".join(sorted(set(unsupported_two_qubit))))
    if measurement_count != 1:
        violations.append(f"measurements={measurement_count}")

    return {
        "rules_respected": len(violations) == 0,
        "rule_violation": "no" if not violations else ";".join(violations),
        "depth": depth,
        "two_qubit_gates": two_qubit_count,
        "gate_counts": dict(ops),
    }


# -----------------------------------------------------------------------------
# PennyLane circuit factory
# -----------------------------------------------------------------------------

def make_pennylane_functions(cfg: ExperimentConfig):
    """Create PennyLane circuits for a given experiment config."""
    dev_exact = qml.device("default.qubit", wires=NUM_QUBITS)
    dev_shots = qml.device("default.qubit", wires=NUM_QUBITS)

    ### Encode the data directly###
    def encode_data(x):
        # x1 → gate(x1) on qubit 0
        # x2 → gate(x2) on qubit 1
        # ...
        # x8 → gate(x8) on qubit 7
        for i in range(NUM_QUBITS):
            for gate in ENCODING_MODES[cfg.encoding_mode]:
                if gate == "rx":
                    qml.RX(x[i], wires=i)
                elif gate == "ry":
                    qml.RY(x[i], wires=i)
                elif gate == "rz":
                    qml.RZ(x[i], wires=i)
                else:
                    raise ValueError(f"Unknown encoding gate: {gate}")

    # This is direct encoding. No scaling. No normalization. No PCA. No get_angles.

    # can be used before encoding, This starts every qubit in a superposition before data encoding.
    def apply_initial_layer():
        if cfg.use_initial_h:
            for i in range(NUM_QUBITS):
                qml.Hadamard(wires=i)

    def apply_measurement_basis():
        if cfg.measurement_basis == "Z":
            pass
        elif cfg.measurement_basis == "X":
            qml.Hadamard(wires=cfg.measured_qubit)
        else:
            raise ValueError(f"Unknown measurement_basis: {cfg.measurement_basis}")

    ### Define trainable layer ###
    def apply_entanglement(layer_id):
        ent = cfg.entangler

        if ent == "chain":
            for i in range(NUM_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])

        elif ent == "brickwork":
            for i in range(0, NUM_QUBITS - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, NUM_QUBITS - 1, 2):
                qml.CNOT(wires=[i, i + 1])

        elif ent == "chain_alternating":
            if layer_id % 2 == 0:
                for i in range(NUM_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
            else:
                for i in reversed(range(NUM_QUBITS - 1)):
                    qml.CNOT(wires=[i + 1, i])

        elif ent == "brickwork_alternating":
            if layer_id % 2 == 0:
                for i in range(0, NUM_QUBITS - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, NUM_QUBITS - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            else:
                # Odd layers: right-to-left brickwork
                for i in reversed(range(1, NUM_QUBITS - 1, 2)):
                    qml.CNOT(wires=[i + 1, i])
                for i in reversed(range(0, NUM_QUBITS - 1, 2)):
                    qml.CNOT(wires=[i + 1, i])

        elif ent == "cz_chain":
            for i in range(NUM_QUBITS - 1):
                qml.CZ(wires=[i, i + 1])

        elif ent == "cz_brickwork":
            for i in range(0, NUM_QUBITS - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, NUM_QUBITS - 1, 2):
                qml.CZ(wires=[i, i + 1])

        elif ent == "cz_chain_alternating":
            if layer_id % 2 == 0:
                for i in range(NUM_QUBITS - 1):
                    qml.CZ(wires=[i, i + 1])
            else:
                for i in reversed(range(NUM_QUBITS - 1)):
                    qml.CZ(wires=[i + 1, i])

        elif ent == "cz_brickwork_alternating":
            if layer_id % 2 == 0:
                for i in range(0, NUM_QUBITS - 1, 2):
                    qml.CZ(wires=[i, i + 1])
                for i in range(1, NUM_QUBITS - 1, 2):
                    qml.CZ(wires=[i, i + 1])
            else:
                # Odd layers: right-to-left brickwork
                for i in reversed(range(1, NUM_QUBITS - 1, 2)):
                    qml.CZ(wires=[i + 1, i])
                for i in reversed(range(0, NUM_QUBITS - 1, 2)):
                    qml.CZ(wires=[i + 1, i])
        else:
            raise ValueError(f"Unknown entangler: {ent}")

    def trainable_layer(layer_weights, layer_id):
        # Trainable single-qubit gates
        for i in range(NUM_QUBITS):
            for gate_id, gate in enumerate(ROTATION_ORDERS[cfg.rotation_order]):
                if gate == "rx":
                    qml.RX(layer_weights[i, gate_id], wires=i)
                elif gate == "ry":
                    qml.RY(layer_weights[i, gate_id], wires=i)
                elif gate == "rz":
                    qml.RZ(layer_weights[i, gate_id], wires=i)
                else:
                    raise ValueError(f"Unknown rotation gate: {gate}")

        # Entangling gates
        apply_entanglement(layer_id)

    def apply_full_circuit(weights, bias, x):
        apply_initial_layer()

        ## encode_data(x), option to only encode data once.
        if not cfg.reupload_data:
            encode_data(x)

        for layer_id in range(cfg.num_layers):
            if cfg.reupload_data:
                encode_data(x)  # each layer gets the freshly encoded data
            trainable_layer(weights[layer_id], layer_id)

        # In-circuit bias instead of classical + bias
        qml.RY(bias, wires=cfg.measured_qubit)

        # Measurement basis change happens right before reading the measured qubit.
        apply_measurement_basis()

    ## full quantum circuit ##
    @qml.qnode(dev_exact, interface="autograd")
    def circuit_expval(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        # PauliZ expectation near +1 → likely measured as 0
        # PauliZ expectation near -1 → likely measured as 1
        return qml.expval(qml.PauliZ(cfg.measured_qubit))

    @qml.qnode(dev_exact, interface="autograd")
    def circuit_probs_exact(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        return qml.probs(wires=cfg.measured_qubit)

    @qml.qnode(dev_shots, shots=1024)
    def circuit_probs_shots(weights, bias, x):
        apply_full_circuit(weights, bias, x)
        return qml.probs(wires=cfg.measured_qubit)

    return circuit_expval, circuit_probs_exact, circuit_probs_shots


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def prob_class_1_from_z(z) -> float:
    # PauliZ returns z = P(0) - P(1), so P(1) = (1 - z) / 2
    return (1 - z) / 2


def predict_from_probs(probs_1: onp.ndarray) -> onp.ndarray:
    return (probs_1 >= 0.5).astype(int)


def evaluate_probs_from_expval(circuit_expval, weights, bias, X_data) -> onp.ndarray:
    probs = []
    for x in X_data:
        z = circuit_expval(weights, bias, x)
        probs.append(float(prob_class_1_from_z(z)))
    return onp.array(probs)


def evaluate_probs_from_qml_probs(circuit_probs, weights, bias, X_data) -> onp.ndarray:
    probs = []
    for x in X_data:
        p = circuit_probs(weights, bias, x)
        probs.append(float(p[1]))
    return onp.array(probs)


def manual_1024_from_exact_probs(probs_1: onp.ndarray, shots: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    counts_1 = rng.binomial(shots, probs_1)
    return counts_1 / shots


def recalls(y_true, preds) -> Tuple[float, float]:
    vals = recall_score(y_true, preds, average=None, labels=[0, 1], zero_division=0)
    return float(vals[0]), float(vals[1])


def train_one_config(
    cfg: ExperimentConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    X_public,
    y_public,
    shots: int,
    shot_mode: str,
) -> Dict[str, object]:
    """Train one valid combination and return metrics/results."""
    circuit_expval, circuit_probs_exact, circuit_probs_shots = make_pennylane_functions(cfg)

    # Full-training class weights. This is more stable than per-mini-batch weights.
    n0_train = onp.sum(y_train == 0)
    n1_train = onp.sum(y_train == 1)
    total_train = len(y_train)
    class_weight_0 = total_train / (2 * n0_train)
    class_weight_1 = (total_train / (2 * n1_train)) * cfg.class_weight_1_multiplier

    def prob_class_1(weights, bias, x):
        z = circuit_expval(weights, bias, x)
        return prob_class_1_from_z(z)

    #### Define balanced-accuracy-friendly loss #####
    # binary cross entropy
    def weighted_bce_loss(weights, bias, X_batch, y_batch):
        probs = [prob_class_1(weights, bias, x) for x in X_batch]
        probs = np.stack(probs)

        eps = 1e-7
        sample_weights = np.where(y_batch == 1, class_weight_1, class_weight_0)

        loss = -np.mean(
            sample_weights
            * (
                y_batch * np.log(probs + eps)
                + (1 - y_batch) * np.log(1 - probs + eps)
            )
        )
        return loss

    def predict_exact(weights, bias, X_data):
        probs = evaluate_probs_from_expval(circuit_expval, weights, bias, X_data)
        return predict_from_probs(probs), probs

    def validation_balanced_accuracy(weights, bias):
        preds, _ = predict_exact(weights, bias, X_val)
        return balanced_accuracy_score(y_val, preds)

    ### Initialize the parameters ###
    # num_layers × num_qubits × rotations-per-qubit
    # Plus one final bias rotation.
    onp.random.seed(cfg.seed)
    weights = cfg.init_scale * np.array(
        onp.random.randn(cfg.num_layers, NUM_QUBITS, cfg.num_train_rot_per_qubit),
        requires_grad=True,
    )
    bias = np.array(0.0, requires_grad=True)

    ###Train the model ###
    opt = qml.AdamOptimizer(stepsize=cfg.lr)
    rng = onp.random.default_rng(cfg.seed)

    best_bal_acc = 0.0
    best_weights = None
    best_bias = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(cfg.num_epochs):
        batch_idx = rng.choice(len(X_train), size=cfg.batch_size, replace=False)
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        weights, bias = opt.step(
            lambda w, b: weighted_bce_loss(w, b, X_batch, y_batch),
            weights,
            bias,
        )

        epochs_trained += 1
        if (epoch + 1) % 5 == 0:
            val_bal_acc = validation_balanced_accuracy(weights, bias)
            print(f"Epoch {epoch + 1:3d} | Validation balanced accuracy: {val_bal_acc:.4f}")

            if val_bal_acc > best_bal_acc:
                best_bal_acc = float(val_bal_acc)
                best_weights = weights.copy()
                best_bias = bias.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5

            if epochs_without_improvement >= cfg.patience:
                print("Early stopping.")
                break

    if best_weights is None or best_bias is None:
        raise RuntimeError("Training finished without saving a best model.")

    # Exact validation and public metrics.
    val_preds, val_probs = predict_exact(best_weights, best_bias, X_val)
    public_preds, public_probs = predict_exact(best_weights, best_bias, X_public)

    val_bal_acc = float(balanced_accuracy_score(y_val, val_preds))
    public_bal_acc = float(balanced_accuracy_score(y_public, public_preds))
    val_recall_0, val_recall_1 = recalls(y_val, val_preds)
    public_recall_0, public_recall_1 = recalls(y_public, public_preds)

    # 1024-shot evaluation. Manual mode is much faster for sweeping many configs.
    if shot_mode == "none":
        public_bal_acc_1024 = ""
    elif shot_mode == "manual":
        public_probs_shot = manual_1024_from_exact_probs(public_probs, shots=shots, seed=cfg.seed)
        public_preds_1024 = predict_from_probs(public_probs_shot)
        public_bal_acc_1024 = float(balanced_accuracy_score(y_public, public_preds_1024))
    elif shot_mode == "pennylane":
        # Update the shots by rebuilding if needed. The qnode decorator above uses 1024,
        # which is the competition limit. For other shot counts, use manual mode.
        if shots != 1024:
            raise ValueError("PennyLane shot mode in this script is set up for 1024 shots. Use --shot-mode manual for other shot counts.")
        public_probs_shot = evaluate_probs_from_qml_probs(circuit_probs_shots, best_weights, best_bias, X_public)
        public_preds_1024 = predict_from_probs(public_probs_shot)
        public_bal_acc_1024 = float(balanced_accuracy_score(y_public, public_preds_1024))
    else:
        raise ValueError(f"Unknown shot_mode: {shot_mode}")

    return {
        "epochs_run": epochs_trained,
        "best_val_bal_acc": val_bal_acc,
        "val_recall_0": val_recall_0,
        "val_recall_1": val_recall_1,
        "public_bal_acc_exact": public_bal_acc,
        "public_recall_0": public_recall_0,
        "public_recall_1": public_recall_1,
        "public_bal_acc_1024": public_bal_acc_1024,
        "best_weights": best_weights,
        "best_bias": best_bias,
    }


# -----------------------------------------------------------------------------
# Weights/QASM artifact helpers
# -----------------------------------------------------------------------------

def make_weights_json(cfg: ExperimentConfig, best_weights, best_bias) -> Dict[str, float]:
    flat_params: List[float] = []

    for layer_id in range(cfg.num_layers):
        for qubit_id in range(NUM_QUBITS):
            for gate_id in range(cfg.num_train_rot_per_qubit):
                flat_params.append(float(best_weights[layer_id, qubit_id, gate_id]))

    flat_params.append(float(best_bias))

    weights_json = {f"theta_{i}": value for i, value in enumerate(flat_params)}

    expected_num_params = cfg.num_trainable_params
    assert len(weights_json) == expected_num_params, (
        f"weights.json has {len(weights_json)} parameters, but expected {expected_num_params}"
    )
    assert list(weights_json.keys()) == [f"theta_{i}" for i in range(expected_num_params)], "theta keys are not sequential"

    return weights_json


def save_artifacts(
    cfg: ExperimentConfig,
    combo_index: int,
    qasm_text: str,
    weights_json: Dict[str, float],
    artifacts_dir: Path,
) -> str:
    run_dir = artifacts_dir / f"run_{combo_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "classifier.qasm", "w", encoding="utf-8") as f:
        f.write(qasm_text)

    with open(run_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return str(run_dir)


# -----------------------------------------------------------------------------
# Results CSV helpers
# -----------------------------------------------------------------------------

RESULT_COLUMNS = [
    "combo_index",
    "config_key",
    "status",
    "lr",
    "batch_size",
    "epochs_run",
    "num_layers",
    "entanglement",
    "measured_qubit",
    "seed",
    "rotation_order",
    "num_train_rot_per_qubit",
    "best_val_bal_acc",
    "val_recall_0",
    "val_recall_1",
    "public_bal_acc_exact",
    "public_recall_0",
    "public_recall_1",
    "public_bal_acc_1024",
    "encoding_mode",
    "initial_h",
    "measurement_basis",
    "reupload_data",
    "init_scale",
    "class_weight_1_multiplier",
    "depth",
    "two_qubit_gates",
    "num_parameters",
    "gate_counts",
    "rules_respected",
    "rule_violation",
    "shot_mode",
    "artifacts_dir",
    "error",
]


def cfg_base_row(combo_index: int, cfg: ExperimentConfig, shot_mode: str) -> Dict[str, object]:
    return {
        "combo_index": combo_index,
        "config_key": cfg.key(),
        "status": "",
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs_run": "",
        "num_layers": cfg.num_layers,
        "entanglement": cfg.entangler,
        "measured_qubit": cfg.measured_qubit,
        "seed": cfg.seed,
        "rotation_order": cfg.rotation_order,
        "num_train_rot_per_qubit": cfg.num_train_rot_per_qubit,
        "best_val_bal_acc": "",
        "val_recall_0": "",
        "val_recall_1": "",
        "public_bal_acc_exact": "",
        "public_recall_0": "",
        "public_recall_1": "",
        "public_bal_acc_1024": "",
        "encoding_mode": cfg.encoding_mode,
        "initial_h": cfg.use_initial_h,
        "measurement_basis": cfg.measurement_basis,
        "reupload_data": cfg.reupload_data,
        "init_scale": cfg.init_scale,
        "class_weight_1_multiplier": cfg.class_weight_1_multiplier,
        "depth": "",
        "two_qubit_gates": "",
        "num_parameters": cfg.num_trainable_params,
        "gate_counts": "",
        "rules_respected": "",
        "rule_violation": "",
        "shot_mode": shot_mode,
        "artifacts_dir": "",
        "error": "",
    }


def append_result(results_path: Path, row: Dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()

    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})


def load_completed_keys(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path)
    except Exception:
        return set()
    if "config_key" not in df.columns:
        return set()
    return set(str(x) for x in df["config_key"].dropna().values)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run prioritized quantum classifier experiments.")
    parser.add_argument("--train-csv", default="public_train.csv")
    parser.add_argument("--test-csv", default="public_test.csv")
    parser.add_argument("--results-csv", default="results_runner.csv")
    parser.add_argument("--artifacts-dir", default="experiment_artifacts")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this combo_index.")
    parser.add_argument("--max-runs", type=int, default=0, help="If >0, run at most this many combinations.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip configs already present in results CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print/save the plan without training.")
    parser.add_argument("--plan-csv", default="experiment_plan.csv", help="Where to save the dry-run/plan CSV.")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--shot-mode", choices=["manual", "pennylane", "none"], default="manual")
    parser.add_argument("--no-save-artifacts", action="store_true", help="Do not save per-run classifier.qasm/weights.json artifacts.")
    return parser.parse_args()


def save_plan_csv(plan: List[ExperimentConfig], plan_csv: Path) -> None:
    rows = []
    for i, cfg in enumerate(plan):
        d = asdict(cfg)
        d["combo_index"] = i
        d["config_key"] = cfg.key()
        d["num_train_rot_per_qubit"] = cfg.num_train_rot_per_qubit
        d["num_parameters"] = cfg.num_trainable_params
        rows.append(d)
    pd.DataFrame(rows).to_csv(plan_csv, index=False)


def main():
    args = parse_args()

    results_path = Path(args.results_csv)
    artifacts_dir = Path(args.artifacts_dir)
    plan = build_experiment_plan()
    save_plan_csv(plan, Path(args.plan_csv))

    print(f"Total planned combinations: {len(plan)}")
    print(f"Plan saved to: {args.plan_csv}")
    print(f"Results CSV: {results_path}")

    if args.dry_run:
        print("Dry run only. No training performed.")
        for i, cfg in enumerate(plan[:20]):
            print(i, cfg)
        if len(plan) > 20:
            print(f"... {len(plan) - 20} more combinations")
        return

    # Load data.
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values

    # validation uses portion of train data not test. only use test data for final split/check
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    # stratify keeps the class balance similar in the training and validation sets.

    X_public = test_df[FEATURE_COLS].values
    y_public = test_df["label"].values

    completed = load_completed_keys(results_path) if args.skip_completed else set()
    run_count = 0

    for combo_index, cfg in enumerate(plan):
        if combo_index < args.start_index:
            continue
        if args.max_runs and run_count >= args.max_runs:
            break
        if args.skip_completed and cfg.key() in completed:
            print(f"Skipping completed combo {combo_index}: {cfg.key()}")
            continue

        run_count += 1
        row = cfg_base_row(combo_index, cfg, args.shot_mode)

        print("\n" + "=" * 80)
        print(f"Running combo {combo_index} | run_count={run_count}")
        print(cfg)

        try:
            # Build QASM first and check rules before spending time training.
            qc, theta, x_params = build_qiskit_classifier(cfg)
            qasm_text = qasm3_with_angle_inputs(qc)
            checks = check_competition_constraints(qc)

            row["depth"] = checks["depth"]
            row["two_qubit_gates"] = checks["two_qubit_gates"]
            row["gate_counts"] = json.dumps(checks["gate_counts"], sort_keys=True)
            row["rules_respected"] = "yes" if checks["rules_respected"] else "no"
            row["rule_violation"] = checks["rule_violation"]

            if not checks["rules_respected"]:
                row["status"] = "rule_violation"
                print(f"Skipping training because rules were not respected: {checks['rule_violation']}")
                append_result(results_path, row)
                continue

            # Train only if the circuit passes competition checks.
            metrics = train_one_config(
                cfg,
                X_train,
                y_train,
                X_val,
                y_val,
                X_public,
                y_public,
                shots=args.shots,
                shot_mode=args.shot_mode,
            )

            row.update({
                "status": "trained",
                "epochs_run": metrics["epochs_run"],
                "best_val_bal_acc": metrics["best_val_bal_acc"],
                "val_recall_0": metrics["val_recall_0"],
                "val_recall_1": metrics["val_recall_1"],
                "public_bal_acc_exact": metrics["public_bal_acc_exact"],
                "public_recall_0": metrics["public_recall_0"],
                "public_recall_1": metrics["public_recall_1"],
                "public_bal_acc_1024": metrics["public_bal_acc_1024"],
            })

            # Save classifier.qasm and weights.json for this run.
            if not args.no_save_artifacts:
                weights_json = make_weights_json(cfg, metrics["best_weights"], metrics["best_bias"])
                run_dir = save_artifacts(cfg, combo_index, qasm_text, weights_json, artifacts_dir)
                row["artifacts_dir"] = run_dir

            print(
                f"Combo {combo_index} done | "
                f"val={row['best_val_bal_acc']} | "
                f"public={row['public_bal_acc_exact']} | "
                f"shot={row['public_bal_acc_1024']} | "
                f"depth={row['depth']} | twoq={row['two_qubit_gates']}"
            )
            append_result(results_path, row)

        except Exception as exc:
            row["status"] = "error"
            row["rules_respected"] = "unknown"
            row["rule_violation"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            append_result(results_path, row)
            print("ERROR in combo", combo_index)
            traceback.print_exc()

    print("\nFinished requested run range.")
    print(f"Results saved to: {results_path}")
    print(f"Artifacts directory: {artifacts_dir}")


if __name__ == "__main__":
    main()
