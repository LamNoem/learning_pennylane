"""
quantum_architecture_runner.py

A separate, compliance-first architecture runner for the 2026 3rd Global
Quantum AI Competition, Challenge 1.

This file is inspired by quantum_experiment_runner.py, but it uses a shared
primitive-operation representation so that:

1. The PennyLane training circuit and exported OpenQASM 3.0 circuit are built
   from the exact same ordered operation list.
2. Every non-native interaction (for example RZZ) is explicitly decomposed
   into the competition's allowed primitive gates before resource counting.
3. Every planned circuit is checked before any training begins.
4. The classification threshold is selected using only the validation split
   of public_train.csv and is saved in weights.json.
5. public_test.csv is used only for post-training evaluation, never for
   optimization, early stopping, threshold selection, or parameter selection.

Competition constraints enforced by this runner
------------------------------------------------
- 2 to 8 qubits; this runner uses exactly 8.
- Circuit depth <= 50, including the final one-qubit measurement.
- At least one CX or CZ.
- At most 80 two-qubit gates.
- Allowed gates only: X, Y, Z, H, S, T, RX, RY, RZ, CX, CZ.
- Exactly one measured qubit and one classical output bit.
- Shots must be between 1 and 1024.
- Raw x1...x8 CSV values are inserted directly as x_0...x_7 rotation angles.
- No scaling, normalization, PCA, feature products, data augmentation, or
  public-test training is performed.

Planned architecture families
-----------------------------
0. Current four-layer brickwork baseline + learned readout head.
1. Four-layer inward TTN, RY_RX_RZ local rotations.
2. Four-layer inward TTN, RY_RZ_RX local rotations.
3. Four-layer inward TTN with two local rotations, RY_RX.
4. Six-layer readout-directed butterfly with RY_RX.
5. Six-layer readout-directed butterfly with RY_RX_RZ.
6. Four-layer directed ring/funnel circuit.
7. Five-layer butterfly with shared trainable RZZ interactions.
8. Five-layer butterfly with separate trainable RZZ interactions.
9. Shared-parameter QCNN-style 8 -> 4 -> 2 -> 1 hierarchy.
10. Two-CX-per-feature MPS/readout-bus circuit.
11. Four-layer phase/IQP-style circuit with trainable RZZ interactions.

Example usage
-------------
Preflight every planned circuit without needing PennyLane:
    python quantum_architecture_runner.py --dry-run

Run all experiments:
    python quantum_architecture_runner.py

Run the first three experiments:
    python quantum_architecture_runner.py --max-runs 3

Run selected named experiments:
    python quantum_architecture_runner.py --only ttn_ry_rx_rz,phase_rzz

Resume while skipping completed configurations:
    python quantum_architecture_runner.py --skip-completed

Use repeated manual 1024-shot simulation for stable reporting:
    python quantum_architecture_runner.py --shot-mode manual --shot-repeats 10

The generated per-run directory contains:
- classifier.qasm
- weights.json, including threshold
- measured_qubit.txt
- encoding_comment.txt
- config.json
- compliance.json
- metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as onp
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Competition constants
# -----------------------------------------------------------------------------

NUM_QUBITS = 8
FEATURE_COLS = [f"x{i}" for i in range(1, NUM_QUBITS + 1)]
LABEL_COL = "label"

MIN_QUBITS = 2
MAX_QUBITS = 8
MAX_DEPTH = 50
MAX_TWO_QUBIT_GATES = 80
MAX_SHOTS = 1024

ALLOWED_GATES = {
    "x", "y", "z",
    "h", "s", "t",
    "rx", "ry", "rz",
    "cx", "cz",
    "measure",
}

ROTATION_ORDERS: Dict[str, Tuple[str, ...]] = {
    "RY_RX": ("ry", "rx"),
    "RY_RZ": ("ry", "rz"),
    "RX_RZ": ("rx", "rz"),
    "RZ_RX": ("rz", "rx"),
    "RY_RX_RZ": ("ry", "rx", "rz"),
    "RY_RZ_RX": ("ry", "rz", "rx"),
}

ENCODING_GATES: Dict[str, Tuple[str, ...]] = {
    "RY": ("ry",),
    "RX": ("rx",),
    "RZ": ("rz",),
    "RY_RZ": ("ry", "rz"),
}

ARCHITECTURES = {
    "brickwork_readout",
    "ttn",
    "butterfly",
    "ring_funnel",
    "rzz_butterfly",
    "qcnn",
    "mps_bus",
    "phase_rzz",
}


# -----------------------------------------------------------------------------
# Primitive program representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterRef:
    """Reference to a raw feature input or trained parameter."""

    kind: str  # "x" or "theta"
    index: int
    scale: float = 1.0

    def qasm(self) -> str:
        if self.kind not in {"x", "theta"}:
            raise ValueError(f"Unsupported parameter kind: {self.kind}")
        base = f"{self.kind}_{self.index}"
        if self.scale == 1.0:
            return base
        if self.scale == -1.0:
            return f"-{base}"
        return f"({self.scale:.17g} * {base})"


@dataclass(frozen=True)
class PrimitiveOp:
    """One already-decomposed competition-primitive operation."""

    gate: str
    wires: Tuple[int, ...]
    parameter: Optional[ParameterRef] = None


@dataclass
class CircuitProgram:
    """Compiled architecture shared by training, QASM export, and rule checks."""

    name: str
    description: str
    ops: List[PrimitiveOp]
    theta_roles: List[str]
    measured_qubit: int
    logical_layers: int
    encoding_comment: str

    @property
    def num_trainable_params(self) -> int:
        return len(self.theta_roles)


class ProgramBuilder:
    """Build a circuit using only final competition primitive gates."""

    def __init__(self) -> None:
        self.ops: List[PrimitiveOp] = []
        self.theta_roles: List[str] = []

    def new_theta(self, role: str) -> int:
        index = len(self.theta_roles)
        self.theta_roles.append(role)
        return index

    def fixed(self, gate: str, *wires: int) -> None:
        self.ops.append(PrimitiveOp(gate=gate, wires=tuple(wires)))

    def data(self, gate: str, wire: int, feature_index: int) -> None:
        self.ops.append(
            PrimitiveOp(
                gate=gate,
                wires=(wire,),
                parameter=ParameterRef("x", feature_index),
            )
        )

    def trainable(
        self,
        gate: str,
        wires: Sequence[int],
        role: str,
        theta_index: Optional[int] = None,
        scale: float = 1.0,
    ) -> int:
        if theta_index is None:
            theta_index = self.new_theta(role)
        self.ops.append(
            PrimitiveOp(
                gate=gate,
                wires=tuple(wires),
                parameter=ParameterRef("theta", theta_index, scale=scale),
            )
        )
        return theta_index

    def measure(self, wire: int) -> None:
        self.ops.append(PrimitiveOp(gate="measure", wires=(wire,)))


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    architecture: str
    description: str
    num_layers: int
    rotation_order: str
    encoding_mode: str
    measured_qubit: int = 0
    use_initial_h: bool = False
    reupload_data: bool = True
    shared_entanglers: bool = False
    shared_qcnn: bool = False
    learned_readout: bool = True
    feature_order: Tuple[int, ...] = tuple(range(NUM_QUBITS))

    lr: float = 0.03
    batch_size: int = 64
    num_epochs: int = 200
    patience: int = 50
    validation_interval: int = 5
    seed: int = 0
    split_seed: int = 0
    init_scale: float = 0.01
    entangler_init_scale: float = 0.0
    class_weight_1_multiplier: float = 0.95

    def key(self) -> str:
        fields = asdict(self)
        return "|".join(f"{key}={fields[key]}" for key in sorted(fields))


# -----------------------------------------------------------------------------
# Shared circuit-building helpers
# -----------------------------------------------------------------------------

TTN_STAGES: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((1, 0), (3, 2), (5, 4), (7, 6)),
    ((2, 0), (6, 4)),
    ((4, 0),),
)

BUTTERFLY_PATTERNS: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((1, 0), (3, 2), (5, 4), (7, 6)),
    ((2, 0), (3, 1), (6, 4), (7, 5)),
    ((4, 0), (5, 1), (6, 2), (7, 3)),
)

RING_FUNNEL_STAGES: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((1, 0), (3, 2), (5, 4), (7, 6)),
    ((2, 1), (4, 3), (6, 5), (7, 0)),
)

QCNN_LEVELS: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),
    ((0, 2), (4, 6)),
    ((0, 4),),
)


def validate_config_shape(cfg: ExperimentConfig) -> None:
    if cfg.architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
    if cfg.rotation_order not in ROTATION_ORDERS:
        raise ValueError(f"Unknown rotation_order: {cfg.rotation_order}")
    if cfg.encoding_mode not in ENCODING_GATES:
        raise ValueError(f"Unknown encoding_mode: {cfg.encoding_mode}")
    if cfg.measured_qubit < 0 or cfg.measured_qubit >= NUM_QUBITS:
        raise ValueError(f"Invalid measured qubit: {cfg.measured_qubit}")
    if sorted(cfg.feature_order) != list(range(NUM_QUBITS)):
        raise ValueError("feature_order must be a permutation of 0..7")
    if cfg.num_layers < 1:
        raise ValueError("num_layers must be positive")
    if cfg.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if cfg.validation_interval < 1:
        raise ValueError("validation_interval must be positive")


def add_direct_encoding(
    builder: ProgramBuilder,
    mode: str,
    feature_order: Sequence[int],
    active_wires: Optional[Sequence[int]] = None,
) -> None:
    """Insert raw x_i values directly as gate angles. No preprocessing."""
    wires = list(range(NUM_QUBITS)) if active_wires is None else list(active_wires)
    for wire in wires:
        feature_index = feature_order[wire]
        for gate in ENCODING_GATES[mode]:
            builder.data(gate, wire, feature_index)


def add_local_rotations(
    builder: ProgramBuilder,
    wires: Sequence[int],
    rotation_order: str,
    role_prefix: str,
) -> None:
    for wire in wires:
        for gate in ROTATION_ORDERS[rotation_order]:
            builder.trainable(gate, (wire,), role=f"{role_prefix}:{gate}:q{wire}")


def add_cx_edges(builder: ProgramBuilder, edges: Sequence[Tuple[int, int]]) -> None:
    for control, target in edges:
        builder.fixed("cx", control, target)


def add_rzz_edges(
    builder: ProgramBuilder,
    edges: Sequence[Tuple[int, int]],
    role_prefix: str,
    shared: bool,
) -> None:
    """
    Add trainable RZZ interactions using the allowed decomposition:

        CX(control, target)
        RZ(phi) on target
        CX(control, target)

    The final submitted circuit therefore contains only CX and RZ.
    """
    shared_index: Optional[int] = None
    if shared:
        shared_index = builder.new_theta(f"entangler:{role_prefix}:shared_rzz")

    for edge_index, (control, target) in enumerate(edges):
        builder.fixed("cx", control, target)
        builder.trainable(
            "rz",
            (target,),
            role=f"entangler:{role_prefix}:edge{edge_index}:rzz",
            theta_index=shared_index,
        )
        builder.fixed("cx", control, target)


def add_learned_readout(builder: ProgramBuilder, measured_qubit: int) -> None:
    """
    Learned one-qubit measurement head.

    RZ is intentionally placed before RY. A final RZ immediately before a Z
    measurement would commute with that measurement and would be redundant.
    """
    builder.trainable("rz", (measured_qubit,), role="readout:azimuth")
    builder.trainable("ry", (measured_qubit,), role="readout:polar_bias")


def encoding_comment_for(cfg: ExperimentConfig) -> str:
    mapping = ", ".join(f"x_{cfg.feature_order[q]}->q{q}" for q in range(NUM_QUBITS))

    if cfg.architecture in {"qcnn", "mps_bus"}:
        frequency = "once at the start of the circuit"
    else:
        frequency = "before every variational block"

    if cfg.architecture == "phase_rzz":
        return (
            "Each raw CSV feature is used directly with no preprocessing: CSV x1..x8 "
            "correspond to symbolic x_0..x_7. An H gate first creates superposition, "
            f"then each x_i is inserted directly as an RZ angle {frequency}. Mapping: {mapping}."
        )

    return (
        "Each raw CSV feature is used directly with no preprocessing: CSV x1..x8 "
        "correspond to symbolic x_0..x_7, and each x_i is inserted directly as an "
        f"{cfg.encoding_mode} rotation angle {frequency}. Mapping: {mapping}."
    )


# -----------------------------------------------------------------------------
# Architecture compilers
# -----------------------------------------------------------------------------


def compile_brickwork_readout(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    for layer in range(cfg.num_layers):
        if cfg.reupload_data:
            add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"local:l{layer}",
        )

        if layer % 2 == 0:
            add_cx_edges(builder, ((0, 1), (2, 3), (4, 5), (6, 7)))
            add_cx_edges(builder, ((1, 2), (3, 4), (5, 6)))
        else:
            add_cx_edges(builder, ((6, 5), (4, 3), (2, 1)))
            add_cx_edges(builder, ((7, 6), (5, 4), (3, 2), (1, 0)))


def compile_ttn(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    for layer in range(cfg.num_layers):
        if cfg.reupload_data:
            add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"local:l{layer}",
        )
        for stage in TTN_STAGES:
            add_cx_edges(builder, stage)


def compile_butterfly(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    for layer in range(cfg.num_layers):
        if cfg.reupload_data:
            add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"local:l{layer}",
        )
        add_cx_edges(builder, BUTTERFLY_PATTERNS[layer % len(BUTTERFLY_PATTERNS)])


def compile_ring_funnel(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    for layer in range(cfg.num_layers):
        if cfg.reupload_data:
            add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"local:l{layer}",
        )
        for stage in RING_FUNNEL_STAGES:
            add_cx_edges(builder, stage)


def compile_rzz_butterfly(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    for layer in range(cfg.num_layers):
        if cfg.reupload_data:
            add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"local:l{layer}",
        )
        add_rzz_edges(
            builder,
            BUTTERFLY_PATTERNS[layer % len(BUTTERFLY_PATTERNS)],
            role_prefix=f"l{layer}",
            shared=cfg.shared_entanglers,
        )


def compile_qcnn(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    # Directly encode all eight raw features exactly once.
    add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)

    for level, pairs in enumerate(QCNN_LEVELS):
        # Five shared parameters per level by default. Sharing reduces capacity
        # and gives the circuit a convolution-like inductive bias.
        shared_indices: Optional[List[int]] = None
        if cfg.shared_qcnn:
            shared_indices = [
                builder.new_theta(f"qcnn:l{level}:shared:{name}")
                for name in ("parent_ry", "parent_rz", "child_ry", "child_rz", "pool_ry")
            ]

        for pair_index, (parent, child) in enumerate(pairs):
            if shared_indices is None:
                indices = [
                    builder.new_theta(f"qcnn:l{level}:p{pair_index}:{name}")
                    for name in ("parent_ry", "parent_rz", "child_ry", "child_rz", "pool_ry")
                ]
            else:
                indices = shared_indices

            builder.trainable(
                "ry", (parent,), role="qcnn:parent_ry", theta_index=indices[0]
            )
            builder.trainable(
                "rz", (parent,), role="qcnn:parent_rz", theta_index=indices[1]
            )
            builder.trainable(
                "ry", (child,), role="qcnn:child_ry", theta_index=indices[2]
            )
            builder.trainable(
                "rz", (child,), role="qcnn:child_rz", theta_index=indices[3]
            )

            # Pool information from child toward parent. This is already
            # expressed only with allowed gates.
            builder.fixed("cx", child, parent)
            builder.trainable(
                "ry", (parent,), role="qcnn:pool_ry", theta_index=indices[4]
            )
            builder.fixed("cx", child, parent)


def compile_mps_bus(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    # Encode all raw features once. q0 is the memory/readout qubit.
    add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)

    # feature_order controls which raw feature is encoded on each physical wire;
    # the readout-bus interaction itself proceeds through physical wires 1..7.
    order = [wire for wire in range(NUM_QUBITS) if wire != cfg.measured_qubit]

    for step, wire in enumerate(order):
        builder.trainable("ry", (wire,), role=f"mps:s{step}:input_ry")
        builder.trainable(
            "ry", (cfg.measured_qubit,), role=f"mps:s{step}:memory_pre_ry"
        )
        builder.fixed("cx", wire, cfg.measured_qubit)
        builder.trainable(
            "rz", (cfg.measured_qubit,), role=f"mps:s{step}:interaction_rz"
        )
        builder.fixed("cx", wire, cfg.measured_qubit)
        builder.trainable(
            "ry", (cfg.measured_qubit,), role=f"mps:s{step}:memory_post_ry"
        )


def compile_phase_rzz(cfg: ExperimentConfig, builder: ProgramBuilder) -> None:
    # RZ encoding on |0> would contribute only a global phase, so H is required.
    for wire in range(NUM_QUBITS):
        builder.fixed("h", wire)

    for layer in range(cfg.num_layers):
        add_direct_encoding(builder, "RZ", cfg.feature_order)
        add_local_rotations(
            builder,
            range(NUM_QUBITS),
            cfg.rotation_order,
            role_prefix=f"phase_local:l{layer}",
        )
        add_rzz_edges(
            builder,
            BUTTERFLY_PATTERNS[layer % len(BUTTERFLY_PATTERNS)],
            role_prefix=f"phase_l{layer}",
            shared=cfg.shared_entanglers,
        )


def compile_program(cfg: ExperimentConfig) -> CircuitProgram:
    validate_config_shape(cfg)
    builder = ProgramBuilder()

    if cfg.use_initial_h and cfg.architecture != "phase_rzz":
        for wire in range(NUM_QUBITS):
            builder.fixed("h", wire)

    if not cfg.reupload_data and cfg.architecture not in {"qcnn", "mps_bus", "phase_rzz"}:
        add_direct_encoding(builder, cfg.encoding_mode, cfg.feature_order)

    compilers = {
        "brickwork_readout": compile_brickwork_readout,
        "ttn": compile_ttn,
        "butterfly": compile_butterfly,
        "ring_funnel": compile_ring_funnel,
        "rzz_butterfly": compile_rzz_butterfly,
        "qcnn": compile_qcnn,
        "mps_bus": compile_mps_bus,
        "phase_rzz": compile_phase_rzz,
    }
    compilers[cfg.architecture](cfg, builder)

    if cfg.learned_readout:
        add_learned_readout(builder, cfg.measured_qubit)
    else:
        builder.trainable("ry", (cfg.measured_qubit,), role="readout:bias")

    builder.measure(cfg.measured_qubit)

    return CircuitProgram(
        name=cfg.name,
        description=cfg.description,
        ops=builder.ops,
        theta_roles=builder.theta_roles,
        measured_qubit=cfg.measured_qubit,
        logical_layers=cfg.num_layers,
        encoding_comment=encoding_comment_for(cfg),
    )


# -----------------------------------------------------------------------------
# Prioritized experiment plan
# -----------------------------------------------------------------------------


def build_experiment_plan() -> List[ExperimentConfig]:
    common = dict(
        measured_qubit=0,
        lr=0.03,
        batch_size=64,
        num_epochs=200,
        patience=50,
        validation_interval=5,
        seed=0,
        split_seed=0,
        init_scale=0.01,
        entangler_init_scale=0.0,
        class_weight_1_multiplier=0.95,
        learned_readout=True,
        feature_order=tuple(range(NUM_QUBITS)),
    )

    return [
        ExperimentConfig(
            name="baseline_readout_head",
            architecture="brickwork_readout",
            description="Run-24-style four-layer brickwork with a learned two-angle readout head.",
            num_layers=4,
            rotation_order="RY_RX_RZ",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="ttn_ry_rx_rz",
            architecture="ttn",
            description="Four-layer inward binary-tree funnel toward measured qubit q0.",
            num_layers=4,
            rotation_order="RY_RX_RZ",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="ttn_ry_rz_rx",
            architecture="ttn",
            description="Four-layer inward TTN using the former best RY_RZ_RX local order.",
            num_layers=4,
            rotation_order="RY_RZ_RX",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="ttn_two_rot",
            architecture="ttn",
            description="Lower-capacity four-layer inward TTN with two local rotations.",
            num_layers=4,
            rotation_order="RY_RX",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="butterfly_six_two_rot",
            architecture="butterfly",
            description="Six-layer long-range butterfly with only four CX gates per block.",
            num_layers=6,
            rotation_order="RY_RX",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="butterfly_six_three_rot",
            architecture="butterfly",
            description="Six-layer long-range butterfly with RY_RX_RZ local rotations.",
            num_layers=6,
            rotation_order="RY_RX_RZ",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="ring_funnel",
            architecture="ring_funnel",
            description="Four-layer two-stage directed ring that repeatedly routes information toward q0.",
            num_layers=4,
            rotation_order="RY_RX_RZ",
            encoding_mode="RY",
            reupload_data=True,
            **common,
        ),
        ExperimentConfig(
            name="rzz_butterfly_shared",
            architecture="rzz_butterfly",
            description="Five-layer butterfly with one identity-initialized shared RZZ angle per layer.",
            num_layers=5,
            rotation_order="RY_RX",
            encoding_mode="RY",
            reupload_data=True,
            shared_entanglers=True,
            **common,
        ),
        ExperimentConfig(
            name="rzz_butterfly_separate",
            architecture="rzz_butterfly",
            description="Five-layer butterfly with separate identity-initialized RZZ angles per edge.",
            num_layers=5,
            rotation_order="RY_RX",
            encoding_mode="RY",
            reupload_data=True,
            shared_entanglers=False,
            **common,
        ),
        ExperimentConfig(
            name="qcnn_shared",
            architecture="qcnn",
            description="Shared-parameter QCNN-style 8-to-4-to-2-to-1 hierarchy.",
            num_layers=3,
            rotation_order="RY_RZ",
            encoding_mode="RY",
            reupload_data=False,
            shared_qcnn=True,
            **common,
        ),
        ExperimentConfig(
            name="mps_bus",
            architecture="mps_bus",
            description="Sequential two-CX readout-bus circuit that feeds q1..q7 into q0.",
            num_layers=7,
            rotation_order="RY_RZ",
            encoding_mode="RY",
            reupload_data=False,
            **common,
        ),
        ExperimentConfig(
            name="phase_rzz",
            architecture="phase_rzz",
            description="Four-layer H+RZ phase encoding with butterfly RZZ interactions.",
            num_layers=4,
            rotation_order="RY_RX",
            encoding_mode="RZ",
            use_initial_h=True,
            reupload_data=True,
            shared_entanglers=False,
            **common,
        ),
    ]


# -----------------------------------------------------------------------------
# Exact primitive resource checks
# -----------------------------------------------------------------------------


def primitive_depth(ops: Sequence[PrimitiveOp], num_qubits: int) -> int:
    """Compute gate depth by greedily parallelizing operations on disjoint wires."""
    wire_depth = [0] * num_qubits
    overall = 0

    for op in ops:
        if not op.wires:
            continue
        op_depth = max(wire_depth[wire] for wire in op.wires) + 1
        for wire in op.wires:
            wire_depth[wire] = op_depth
        overall = max(overall, op_depth)

    return overall


def check_competition_constraints(
    program: CircuitProgram,
    shots: int,
) -> Dict[str, object]:
    gate_counts = Counter(op.gate for op in program.ops)
    bad_gates = sorted(set(gate_counts) - ALLOWED_GATES)
    two_qubit_ops = [op for op in program.ops if len(op.wires) == 2]
    unsupported_two_qubit = sorted(
        {op.gate for op in two_qubit_ops if op.gate not in {"cx", "cz"}}
    )
    two_qubit_count = sum(1 for op in two_qubit_ops if op.gate in {"cx", "cz"})
    measurements = [op for op in program.ops if op.gate == "measure"]
    measured_wires = sorted({op.wires[0] for op in measurements})
    depth = primitive_depth(program.ops, NUM_QUBITS)

    x_refs = {
        op.parameter.index
        for op in program.ops
        if op.parameter is not None and op.parameter.kind == "x"
    }
    theta_refs = {
        op.parameter.index
        for op in program.ops
        if op.parameter is not None and op.parameter.kind == "theta"
    }
    invalid_parameterized_gates = sorted({
        op.gate
        for op in program.ops
        if op.parameter is not None and op.gate not in {"rx", "ry", "rz"}
    })
    gate_after_measurement = False
    seen_measurement = False
    for op in program.ops:
        if op.gate == "measure":
            seen_measurement = True
        elif seen_measurement:
            gate_after_measurement = True

    violations: List[str] = []
    if NUM_QUBITS < MIN_QUBITS:
        violations.append(f"qubits<{MIN_QUBITS}")
    if NUM_QUBITS > MAX_QUBITS:
        violations.append(f"qubits>{MAX_QUBITS}")
    if program.logical_layers > MAX_DEPTH:
        violations.append(f"logical_layers>{MAX_DEPTH}")
    if depth > MAX_DEPTH:
        violations.append(f"depth>{MAX_DEPTH}")
    if two_qubit_count < 1:
        violations.append("no_entangling_gate")
    if two_qubit_count > MAX_TWO_QUBIT_GATES:
        violations.append(f"two_qubit>{MAX_TWO_QUBIT_GATES}")
    if bad_gates:
        violations.append("bad_gates:" + "+".join(bad_gates))
    if unsupported_two_qubit:
        violations.append(
            "unsupported_two_qubit:" + "+".join(unsupported_two_qubit)
        )
    if len(measurements) != 1:
        violations.append(f"measurements={len(measurements)}")
    if measured_wires != [program.measured_qubit]:
        violations.append(f"measured_wires={measured_wires}")
    if gate_after_measurement:
        violations.append("gate_after_measurement")
    if x_refs != set(range(NUM_QUBITS)):
        violations.append(f"x_refs={sorted(x_refs)}")
    if theta_refs != set(range(program.num_trainable_params)):
        violations.append(f"theta_refs={sorted(theta_refs)}")
    if invalid_parameterized_gates:
        violations.append(
            "invalid_parameterized_gates:" + "+".join(invalid_parameterized_gates)
        )
    if shots < 1 or shots > MAX_SHOTS:
        violations.append(f"shots={shots}")

    return {
        "rules_respected": not violations,
        "rule_violation": "no" if not violations else ";".join(violations),
        "num_qubits": NUM_QUBITS,
        "logical_layers": program.logical_layers,
        "depth": depth,
        "two_qubit_gates": two_qubit_count,
        "measurement_count": len(measurements),
        "measured_qubit": program.measured_qubit,
        "num_trainable_parameters": program.num_trainable_params,
        "gate_counts": dict(sorted(gate_counts.items())),
        "shots": shots,
        "decomposition_note": (
            "All operations in this report are final allowed primitives. "
            "RZZ interactions were decomposed to CX-RZ-CX before counting."
        ),
    }


def preflight_plan(
    plan: Sequence[ExperimentConfig],
    shots: int,
) -> List[Tuple[ExperimentConfig, CircuitProgram, Dict[str, object]]]:
    compiled = []
    violations = []

    for index, cfg in enumerate(plan):
        program = compile_program(cfg)
        checks = check_competition_constraints(program, shots=shots)
        compiled.append((cfg, program, checks))
        if not checks["rules_respected"]:
            violations.append((index, cfg.name, checks["rule_violation"]))

    if violations:
        text = "\n".join(
            f"  combo {index} {name}: {reason}"
            for index, name, reason in violations
        )
        raise RuntimeError(
            "The experiment plan contains non-compliant circuits. Training was "
            f"aborted before reading data:\n{text}"
        )

    return compiled


# -----------------------------------------------------------------------------
# OpenQASM 3.0 generation
# -----------------------------------------------------------------------------


def program_to_qasm3(program: CircuitProgram) -> str:
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        "// Raw feature inputs: x_0 corresponds to CSV x1, ..., x_7 to CSV x8.",
    ]

    for index in range(NUM_QUBITS):
        lines.append(f"input angle[64] x_{index};")

    lines.append("")
    lines.append("// Trained parameters supplied through weights.json.")
    for index in range(program.num_trainable_params):
        lines.append(f"input angle[64] theta_{index};")

    lines.extend([
        "",
        "bit[1] c;",
        f"qubit[{NUM_QUBITS}] q;",
        "",
        f"// {program.encoding_comment}",
    ])

    for op in program.ops:
        if op.gate == "measure":
            lines.append(f"c[0] = measure q[{op.wires[0]}];")
            continue

        qargs = ", ".join(f"q[{wire}]" for wire in op.wires)
        if op.parameter is None:
            lines.append(f"{op.gate} {qargs};")
        else:
            lines.append(f"{op.gate}({op.parameter.qasm()}) {qargs};")

    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# PennyLane execution generated from the same primitive program
# -----------------------------------------------------------------------------


def make_pennylane_functions(program: CircuitProgram, shots: int, shot_seed: int):
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for training. Install it in the competition "
            "environment, or use --dry-run for compliance preflight only."
        ) from exc

    dev_exact = qml.device("default.qubit", wires=NUM_QUBITS)

    # seed is supported by current default.qubit versions. Fall back cleanly for
    # older PennyLane versions that do not accept it.
    try:
        dev_shots = qml.device(
            "default.qubit", wires=NUM_QUBITS, shots=shots, seed=shot_seed
        )
    except TypeError:
        dev_shots = qml.device("default.qubit", wires=NUM_QUBITS, shots=shots)

    def parameter_value(op: PrimitiveOp, theta, x):
        assert op.parameter is not None
        if op.parameter.kind == "theta":
            value = theta[op.parameter.index]
        elif op.parameter.kind == "x":
            value = x[op.parameter.index]
        else:
            raise ValueError(f"Unknown parameter kind: {op.parameter.kind}")
        return op.parameter.scale * value

    def apply_program(theta, x):
        for op in program.ops:
            gate = op.gate
            if gate == "measure":
                continue

            if op.parameter is None:
                if gate == "x":
                    qml.PauliX(wires=op.wires[0])
                elif gate == "y":
                    qml.PauliY(wires=op.wires[0])
                elif gate == "z":
                    qml.PauliZ(wires=op.wires[0])
                elif gate == "h":
                    qml.Hadamard(wires=op.wires[0])
                elif gate == "s":
                    qml.S(wires=op.wires[0])
                elif gate == "t":
                    qml.T(wires=op.wires[0])
                elif gate == "cx":
                    qml.CNOT(wires=list(op.wires))
                elif gate == "cz":
                    qml.CZ(wires=list(op.wires))
                else:
                    raise ValueError(f"Unsupported fixed gate: {gate}")
            else:
                angle = parameter_value(op, theta, x)
                if gate == "rx":
                    qml.RX(angle, wires=op.wires[0])
                elif gate == "ry":
                    qml.RY(angle, wires=op.wires[0])
                elif gate == "rz":
                    qml.RZ(angle, wires=op.wires[0])
                else:
                    raise ValueError(f"Unsupported parameterized gate: {gate}")

    @qml.qnode(dev_exact, interface="autograd")
    def circuit_expval(theta, x):
        apply_program(theta, x)
        return qml.expval(qml.PauliZ(program.measured_qubit))

    @qml.qnode(dev_exact, interface="autograd")
    def circuit_probs_exact(theta, x):
        apply_program(theta, x)
        return qml.probs(wires=program.measured_qubit)

    @qml.qnode(dev_shots)
    def circuit_probs_shots(theta, x):
        apply_program(theta, x)
        return qml.probs(wires=program.measured_qubit)

    return qml, circuit_expval, circuit_probs_exact, circuit_probs_shots


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def prob_class_1_from_z(z):
    return (1.0 - z) / 2.0


def predict_with_threshold(probs_1: onp.ndarray, threshold: float) -> onp.ndarray:
    return (probs_1 >= threshold).astype(int)


def recalls(y_true, predictions) -> Tuple[float, float]:
    values = recall_score(
        y_true,
        predictions,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )
    return float(values[0]), float(values[1])


def find_validation_threshold(
    y_true: onp.ndarray,
    probs_1: onp.ndarray,
) -> Tuple[float, float]:
    """
    Find the exact piecewise-constant balanced-accuracy optimum on validation.

    Public-test labels are never passed to this function. If several thresholds
    tie, choose the one closest to 0.5 to reduce threshold overfitting.
    """
    unique = onp.unique(onp.asarray(probs_1, dtype=float))
    candidates: List[float] = [0.0, 0.5, 1.0]

    if len(unique) > 1:
        candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())

    # Restrict to a valid probability threshold range and remove duplicates.
    candidates = sorted({float(onp.clip(value, 0.0, 1.0)) for value in candidates})

    best_threshold = 0.5
    best_score = -math.inf
    best_distance = math.inf

    for threshold in candidates:
        predictions = predict_with_threshold(probs_1, threshold)
        score = float(balanced_accuracy_score(y_true, predictions))
        distance = abs(threshold - 0.5)

        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and distance < best_distance
        ):
            best_threshold = threshold
            best_score = score
            best_distance = distance

    return best_threshold, best_score


def evaluate_probs_from_expval(circuit_expval, theta, X_data) -> onp.ndarray:
    probabilities = []
    for x in X_data:
        z_value = circuit_expval(theta, x)
        probabilities.append(float(prob_class_1_from_z(z_value)))
    return onp.asarray(probabilities, dtype=float)


def evaluate_probs_from_qml_probs(circuit_probs, theta, X_data) -> onp.ndarray:
    probabilities = []
    for x in X_data:
        result = circuit_probs(theta, x)
        probabilities.append(float(result[1]))
    return onp.asarray(probabilities, dtype=float)


def balanced_batch_indices(
    y_train: onp.ndarray,
    batch_size: int,
    rng: onp.random.Generator,
) -> onp.ndarray:
    class_0 = onp.flatnonzero(y_train == 0)
    class_1 = onp.flatnonzero(y_train == 1)

    n0 = batch_size // 2
    n1 = batch_size - n0
    if len(class_0) < n0 or len(class_1) < n1:
        raise ValueError(
            "Balanced batch is larger than an available class. Reduce --batch-size "
            "or implement full-dataset training."
        )

    selected_0 = rng.choice(class_0, size=n0, replace=False)
    selected_1 = rng.choice(class_1, size=n1, replace=False)
    selected = onp.concatenate([selected_0, selected_1])
    rng.shuffle(selected)
    return selected


def initialize_theta(program: CircuitProgram, cfg: ExperimentConfig):
    from pennylane import numpy as pnp

    rng = onp.random.default_rng(cfg.seed)
    values = onp.zeros(program.num_trainable_params, dtype=float)

    for index, role in enumerate(program.theta_roles):
        if role.startswith("entangler:"):
            values[index] = cfg.entangler_init_scale * rng.standard_normal()
        elif role.startswith("readout:"):
            values[index] = 0.0
        else:
            values[index] = cfg.init_scale * rng.standard_normal()

    return pnp.array(values, requires_grad=True)


def manual_shot_scores(
    exact_probs: onp.ndarray,
    y_true: onp.ndarray,
    threshold: float,
    shots: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    scores = []
    for repeat in range(repeats):
        rng = onp.random.default_rng(seed + 10_000 + repeat)
        safe_probs = onp.clip(exact_probs, 0.0, 1.0)
        counts_1 = rng.binomial(shots, safe_probs)
        shot_probs = counts_1 / shots
        predictions = predict_with_threshold(shot_probs, threshold)
        scores.append(float(balanced_accuracy_score(y_true, predictions)))

    values = onp.asarray(scores, dtype=float)
    return {
        "shot_bacc_mean": float(values.mean()),
        "shot_bacc_std": float(values.std(ddof=0)),
        "shot_bacc_min": float(values.min()),
        "shot_bacc_max": float(values.max()),
    }


def train_one_config(
    cfg: ExperimentConfig,
    program: CircuitProgram,
    X_train: onp.ndarray,
    y_train: onp.ndarray,
    X_val: onp.ndarray,
    y_val: onp.ndarray,
    X_public: onp.ndarray,
    y_public: onp.ndarray,
    shots: int,
    shot_mode: str,
    shot_repeats: int,
) -> Dict[str, object]:
    qml, circuit_expval, _circuit_probs_exact, circuit_probs_shots = (
        make_pennylane_functions(program, shots=shots, shot_seed=cfg.seed)
    )
    from pennylane import numpy as pnp

    theta = initialize_theta(program, cfg)
    optimizer = qml.AdamOptimizer(stepsize=cfg.lr)
    rng = onp.random.default_rng(cfg.seed)

    def class_balanced_bce(current_theta, X_batch, y_batch):
        probabilities = pnp.stack(
            [prob_class_1_from_z(circuit_expval(current_theta, x)) for x in X_batch]
        )
        y_batch_pnp = pnp.asarray(y_batch)
        eps = 1e-7

        # The batch sampler provides equal class counts. This weighted form avoids
        # boolean indexing through Autograd while preserving the class-1 nudge.
        sample_weights = pnp.where(
            y_batch_pnp == 1, cfg.class_weight_1_multiplier, 1.0
        )
        losses = -(
            y_batch_pnp * pnp.log(probabilities + eps)
            + (1.0 - y_batch_pnp) * pnp.log(1.0 - probabilities + eps)
        )
        return pnp.mean(sample_weights * losses)

    best_val_bacc = -math.inf
    best_theta = None
    best_threshold = 0.5
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(cfg.num_epochs):
        indices = balanced_batch_indices(y_train, cfg.batch_size, rng)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        theta = optimizer.step(
            lambda current_theta: class_balanced_bce(
                current_theta, X_batch, y_batch
            ),
            theta,
        )
        epochs_run += 1

        if (epoch + 1) % cfg.validation_interval == 0:
            val_probs = evaluate_probs_from_expval(circuit_expval, theta, X_val)
            threshold, val_bacc = find_validation_threshold(y_val, val_probs)
            print(
                f"Epoch {epoch + 1:3d} | val_bacc={val_bacc:.4f} "
                f"| threshold={threshold:.5f}"
            )

            if val_bacc > best_val_bacc + 1e-12:
                best_val_bacc = float(val_bacc)
                best_theta = theta.copy()
                best_threshold = float(threshold)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += cfg.validation_interval

            if epochs_without_improvement >= cfg.patience:
                print("Early stopping.")
                break

    if best_theta is None:
        raise RuntimeError("No best parameter vector was saved during training.")

    # Recompute exact metrics from the validation-selected weights and threshold.
    val_probs = evaluate_probs_from_expval(circuit_expval, best_theta, X_val)
    val_predictions = predict_with_threshold(val_probs, best_threshold)
    val_bacc = float(balanced_accuracy_score(y_val, val_predictions))
    val_recall_0, val_recall_1 = recalls(y_val, val_predictions)

    # The public set enters only here, after training and model selection are done.
    public_probs = evaluate_probs_from_expval(circuit_expval, best_theta, X_public)
    public_predictions = predict_with_threshold(public_probs, best_threshold)
    public_bacc = float(balanced_accuracy_score(y_public, public_predictions))
    public_recall_0, public_recall_1 = recalls(y_public, public_predictions)

    shot_metrics = {
        "shot_bacc_mean": "",
        "shot_bacc_std": "",
        "shot_bacc_min": "",
        "shot_bacc_max": "",
    }

    if shot_mode == "manual":
        shot_metrics = manual_shot_scores(
            exact_probs=public_probs,
            y_true=y_public,
            threshold=best_threshold,
            shots=shots,
            repeats=shot_repeats,
            seed=cfg.seed,
        )
    elif shot_mode == "pennylane":
        scores = []
        # Recreate the shot device for independent seeded repeats where supported.
        for repeat in range(shot_repeats):
            _, _, _, repeated_shot_circuit = make_pennylane_functions(
                program,
                shots=shots,
                shot_seed=cfg.seed + 10_000 + repeat,
            )
            shot_probs = evaluate_probs_from_qml_probs(
                repeated_shot_circuit, best_theta, X_public
            )
            predictions = predict_with_threshold(shot_probs, best_threshold)
            scores.append(float(balanced_accuracy_score(y_public, predictions)))
        values = onp.asarray(scores, dtype=float)
        shot_metrics = {
            "shot_bacc_mean": float(values.mean()),
            "shot_bacc_std": float(values.std(ddof=0)),
            "shot_bacc_min": float(values.min()),
            "shot_bacc_max": float(values.max()),
        }
    elif shot_mode != "none":
        raise ValueError(f"Unknown shot mode: {shot_mode}")

    return {
        "epochs_run": epochs_run,
        "best_theta": best_theta,
        "threshold": best_threshold,
        "best_val_bal_acc": val_bacc,
        "val_recall_0": val_recall_0,
        "val_recall_1": val_recall_1,
        "public_bal_acc_exact": public_bacc,
        "public_recall_0": public_recall_0,
        "public_recall_1": public_recall_1,
        **shot_metrics,
    }


# -----------------------------------------------------------------------------
# Artifact helpers
# -----------------------------------------------------------------------------


def make_weights_json(
    program: CircuitProgram,
    theta,
    threshold: float,
) -> Dict[str, float]:
    if len(theta) != program.num_trainable_params:
        raise ValueError(
            f"Parameter mismatch: received {len(theta)}, expected "
            f"{program.num_trainable_params}"
        )

    output = {
        f"theta_{index}": float(theta[index])
        for index in range(program.num_trainable_params)
    }
    output["threshold"] = float(threshold)
    return output


def save_run_artifacts(
    cfg: ExperimentConfig,
    program: CircuitProgram,
    checks: Dict[str, object],
    metrics: Dict[str, object],
    combo_index: int,
    artifacts_root: Path,
) -> str:
    run_dir = artifacts_root / f"run_{combo_index:04d}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    qasm_text = program_to_qasm3(program)
    weights = make_weights_json(
        program,
        metrics["best_theta"],
        float(metrics["threshold"]),
    )

    (run_dir / "classifier.qasm").write_text(qasm_text, encoding="utf-8")
    (run_dir / "weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )
    (run_dir / "measured_qubit.txt").write_text(
        str(program.measured_qubit), encoding="utf-8"
    )
    (run_dir / "encoding_comment.txt").write_text(
        program.encoding_comment, encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (run_dir / "compliance.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "best_theta"
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2), encoding="utf-8"
    )

    return str(run_dir)


# -----------------------------------------------------------------------------
# Results and plan CSV helpers
# -----------------------------------------------------------------------------

RESULT_COLUMNS = [
    "combo_index",
    "name",
    "config_key",
    "status",
    "architecture",
    "description",
    "num_layers",
    "rotation_order",
    "encoding_mode",
    "measured_qubit",
    "reupload_data",
    "shared_entanglers",
    "shared_qcnn",
    "learned_readout",
    "lr",
    "batch_size",
    "epochs_run",
    "seed",
    "split_seed",
    "init_scale",
    "entangler_init_scale",
    "class_weight_1_multiplier",
    "threshold",
    "best_val_bal_acc",
    "val_recall_0",
    "val_recall_1",
    "public_bal_acc_exact",
    "public_recall_0",
    "public_recall_1",
    "shot_bacc_mean",
    "shot_bacc_std",
    "shot_bacc_min",
    "shot_bacc_max",
    "shots",
    "shot_mode",
    "shot_repeats",
    "depth",
    "two_qubit_gates",
    "num_parameters",
    "gate_counts",
    "rules_respected",
    "rule_violation",
    "encoding_comment",
    "artifacts_dir",
    "error",
]


def base_result_row(
    combo_index: int,
    cfg: ExperimentConfig,
    program: CircuitProgram,
    checks: Dict[str, object],
    args,
) -> Dict[str, object]:
    return {
        "combo_index": combo_index,
        "name": cfg.name,
        "config_key": cfg.key(),
        "status": "",
        "architecture": cfg.architecture,
        "description": cfg.description,
        "num_layers": cfg.num_layers,
        "rotation_order": cfg.rotation_order,
        "encoding_mode": cfg.encoding_mode,
        "measured_qubit": cfg.measured_qubit,
        "reupload_data": cfg.reupload_data,
        "shared_entanglers": cfg.shared_entanglers,
        "shared_qcnn": cfg.shared_qcnn,
        "learned_readout": cfg.learned_readout,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs_run": "",
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "init_scale": cfg.init_scale,
        "entangler_init_scale": cfg.entangler_init_scale,
        "class_weight_1_multiplier": cfg.class_weight_1_multiplier,
        "threshold": "",
        "best_val_bal_acc": "",
        "val_recall_0": "",
        "val_recall_1": "",
        "public_bal_acc_exact": "",
        "public_recall_0": "",
        "public_recall_1": "",
        "shot_bacc_mean": "",
        "shot_bacc_std": "",
        "shot_bacc_min": "",
        "shot_bacc_max": "",
        "shots": args.shots,
        "shot_mode": args.shot_mode,
        "shot_repeats": args.shot_repeats,
        "depth": checks["depth"],
        "two_qubit_gates": checks["two_qubit_gates"],
        "num_parameters": program.num_trainable_params,
        "gate_counts": json.dumps(checks["gate_counts"], sort_keys=True),
        "rules_respected": "yes" if checks["rules_respected"] else "no",
        "rule_violation": checks["rule_violation"],
        "encoding_comment": program.encoding_comment,
        "artifacts_dir": "",
        "error": "",
    }


def append_result(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()
    if "config_key" not in frame.columns:
        return set()
    return set(str(value) for value in frame["config_key"].dropna())


def save_plan_csv(
    compiled_plan: Sequence[Tuple[ExperimentConfig, CircuitProgram, Dict[str, object]]],
    path: Path,
) -> None:
    rows = []
    for index, (cfg, program, checks) in enumerate(compiled_plan):
        rows.append({
            "combo_index": index,
            "name": cfg.name,
            "architecture": cfg.architecture,
            "description": cfg.description,
            "num_layers": cfg.num_layers,
            "rotation_order": cfg.rotation_order,
            "encoding_mode": cfg.encoding_mode,
            "reupload_data": cfg.reupload_data,
            "shared_entanglers": cfg.shared_entanglers,
            "shared_qcnn": cfg.shared_qcnn,
            "learned_readout": cfg.learned_readout,
            "num_parameters": program.num_trainable_params,
            "depth": checks["depth"],
            "two_qubit_gates": checks["two_qubit_gates"],
            "gate_counts": json.dumps(checks["gate_counts"], sort_keys=True),
            "rules_respected": checks["rules_respected"],
            "rule_violation": checks["rule_violation"],
            "encoding_comment": program.encoding_comment,
            "config_key": cfg.key(),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Data integrity checks
# -----------------------------------------------------------------------------


def load_data(train_csv: Path, test_csv: Path, split_seed: int):
    train_frame = pd.read_csv(train_csv)
    public_frame = pd.read_csv(test_csv)

    required = set(FEATURE_COLS + [LABEL_COL])
    missing_train = sorted(required - set(train_frame.columns))
    missing_public = sorted(required - set(public_frame.columns))
    if missing_train:
        raise ValueError(f"Training CSV is missing columns: {missing_train}")
    if missing_public:
        raise ValueError(f"Public test CSV is missing columns: {missing_public}")

    X = train_frame[FEATURE_COLS].to_numpy(dtype=float, copy=True)
    y = train_frame[LABEL_COL].to_numpy(dtype=int, copy=True)
    X_public = public_frame[FEATURE_COLS].to_numpy(dtype=float, copy=True)
    y_public = public_frame[LABEL_COL].to_numpy(dtype=int, copy=True)

    if set(onp.unique(y)) - {0, 1}:
        raise ValueError("Training labels must be binary 0/1.")
    if set(onp.unique(y_public)) - {0, 1}:
        raise ValueError("Public labels must be binary 0/1.")
    if not onp.isfinite(X).all() or not onp.isfinite(X_public).all():
        raise ValueError("Feature data contains NaN or infinite values.")

    # The raw arrays are split directly. No feature transformation occurs.
    indices = onp.arange(len(X))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=split_seed,
        stratify=y,
    )

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices],
        X_public,
        y_public,
        train_indices,
        val_indices,
    )


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compliance-first quantum architecture experiments."
    )
    parser.add_argument("--train-csv", default="public_train.csv")
    parser.add_argument("--test-csv", default="public_test.csv")
    parser.add_argument("--results-csv", default="architecture_results.csv")
    parser.add_argument("--plan-csv", default="architecture_plan.csv")
    parser.add_argument("--artifacts-dir", default="architecture_artifacts")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment names to run, preserving plan order.",
    )
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument(
        "--shot-mode",
        choices=["manual", "pennylane", "none"],
        default="manual",
    )
    parser.add_argument("--shot-repeats", type=int, default=5)
    parser.add_argument("--no-save-artifacts", action="store_true")
    return parser.parse_args()


def print_preflight_table(
    compiled_plan: Sequence[Tuple[ExperimentConfig, CircuitProgram, Dict[str, object]]]
) -> None:
    print("\nCompliance preflight")
    print("-" * 112)
    print(
        f"{'idx':>3}  {'name':<29} {'depth':>5} {'2q':>4} "
        f"{'params':>6} {'valid':>6}  description"
    )
    print("-" * 112)
    for index, (cfg, program, checks) in enumerate(compiled_plan):
        print(
            f"{index:>3}  {cfg.name:<29} {checks['depth']:>5} "
            f"{checks['two_qubit_gates']:>4} "
            f"{program.num_trainable_params:>6} "
            f"{str(checks['rules_respected']):>6}  {cfg.description}"
        )
    print("-" * 112)


def print_final_ranking(results_path: Path) -> None:
    if not results_path.exists():
        return
    try:
        frame = pd.read_csv(results_path)
    except Exception:
        return
    trained = frame[frame.get("status", "") == "trained"].copy()
    if trained.empty:
        return

    sort_columns = [
        column
        for column in ["best_val_bal_acc", "shot_bacc_mean", "public_bal_acc_exact"]
        if column in trained.columns
    ]
    trained = trained.sort_values(sort_columns, ascending=False)
    columns = [
        column
        for column in [
            "combo_index",
            "name",
            "best_val_bal_acc",
            "public_bal_acc_exact",
            "shot_bacc_mean",
            "shot_bacc_std",
            "threshold",
            "depth",
            "two_qubit_gates",
        ]
        if column in trained.columns
    ]
    print("\nCurrent validation-first ranking")
    print(trained[columns].head(10).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.shots < 1 or args.shots > MAX_SHOTS:
        raise ValueError(f"--shots must be between 1 and {MAX_SHOTS}.")
    if args.shot_repeats < 1:
        raise ValueError("--shot-repeats must be at least 1.")

    plan = build_experiment_plan()
    selected_names = {
        value.strip() for value in args.only.split(",") if value.strip()
    }
    if selected_names:
        known_names = {cfg.name for cfg in plan}
        unknown = sorted(selected_names - known_names)
        if unknown:
            raise ValueError(f"Unknown --only experiment names: {unknown}")
        plan = [cfg for cfg in plan if cfg.name in selected_names]

    # This strict preflight occurs before data is read or PennyLane is imported.
    compiled_plan = preflight_plan(plan, shots=args.shots)
    save_plan_csv(compiled_plan, Path(args.plan_csv))
    print_preflight_table(compiled_plan)
    print(f"Plan saved to: {args.plan_csv}")

    if args.dry_run:
        print("Dry run complete. Every listed circuit passed the enforced rules.")
        return

    if not compiled_plan:
        print("No configurations selected.")
        return

    # All planned configurations use the same deterministic split seed.
    split_seeds = {cfg.split_seed for cfg, _, _ in compiled_plan}
    if len(split_seeds) != 1:
        raise ValueError("All configurations must share one split_seed for comparison.")
    split_seed = next(iter(split_seeds))

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_public,
        y_public,
        train_indices,
        val_indices,
    ) = load_data(Path(args.train_csv), Path(args.test_csv), split_seed=split_seed)

    artifacts_root = Path(args.artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    onp.save(artifacts_root / "train_indices.npy", train_indices)
    onp.save(artifacts_root / "validation_indices.npy", val_indices)
    (artifacts_root / "data_policy.txt").write_text(
        "Training and early stopping use only the training portion of public_train.csv.\n"
        "Threshold selection uses only the validation portion of public_train.csv.\n"
        "public_test.csv is used only for final reporting after model selection.\n"
        "Raw features are inserted directly as circuit angles without preprocessing.\n"
        "No data augmentation is performed.\n",
        encoding="utf-8",
    )

    results_path = Path(args.results_csv)
    done = completed_keys(results_path) if args.skip_completed else set()
    run_count = 0

    for combo_index, (cfg, program, checks) in enumerate(compiled_plan):
        if combo_index < args.start_index:
            continue
        if args.max_runs and run_count >= args.max_runs:
            break
        if cfg.key() in done:
            print(f"Skipping completed combo {combo_index}: {cfg.name}")
            continue

        run_count += 1
        row = base_result_row(combo_index, cfg, program, checks, args)

        print("\n" + "=" * 88)
        print(f"Running combo {combo_index}: {cfg.name}")
        print(cfg.description)
        print(
            f"depth={checks['depth']} | two_qubit={checks['two_qubit_gates']} "
            f"| parameters={program.num_trainable_params}"
        )

        try:
            metrics = train_one_config(
                cfg=cfg,
                program=program,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_public=X_public,
                y_public=y_public,
                shots=args.shots,
                shot_mode=args.shot_mode,
                shot_repeats=args.shot_repeats,
            )

            row.update({
                "status": "trained",
                "epochs_run": metrics["epochs_run"],
                "threshold": metrics["threshold"],
                "best_val_bal_acc": metrics["best_val_bal_acc"],
                "val_recall_0": metrics["val_recall_0"],
                "val_recall_1": metrics["val_recall_1"],
                "public_bal_acc_exact": metrics["public_bal_acc_exact"],
                "public_recall_0": metrics["public_recall_0"],
                "public_recall_1": metrics["public_recall_1"],
                "shot_bacc_mean": metrics["shot_bacc_mean"],
                "shot_bacc_std": metrics["shot_bacc_std"],
                "shot_bacc_min": metrics["shot_bacc_min"],
                "shot_bacc_max": metrics["shot_bacc_max"],
            })

            if not args.no_save_artifacts:
                row["artifacts_dir"] = save_run_artifacts(
                    cfg=cfg,
                    program=program,
                    checks=checks,
                    metrics=metrics,
                    combo_index=combo_index,
                    artifacts_root=artifacts_root,
                )

            append_result(results_path, row)
            print(
                f"Done | val={row['best_val_bal_acc']:.4f} "
                f"| public={row['public_bal_acc_exact']:.4f} "
                f"| shots_mean={row['shot_bacc_mean']} "
                f"| threshold={row['threshold']:.5f}"
            )

        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            append_result(results_path, row)
            print(f"ERROR in combo {combo_index}: {cfg.name}")
            traceback.print_exc()

    print("\nFinished requested run range.")
    print(f"Results saved to: {results_path}")
    print(f"Artifacts directory: {artifacts_root}")
    print_final_ranking(results_path)


if __name__ == "__main__":
    main()
