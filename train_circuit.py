import pandas as pd
import numpy as onp

import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


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
print(y)
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
num_layers = 4

dev = qml.device("default.qubit", wires=num_qubits)
#We use 8 qubits because there are 8 features

### Encode the data directly### 
def encode_data(x):
    # x1 → RY(x1) on qubit 0
    # x2 → RY(x2) on qubit 1
    # x3 → RY(x3) on qubit 2
    # ...
    # x8 → RY(x8) on qubit 7
    for i in range(num_qubits):
        qml.RY(x[i], wires=i)

#This is direct encoding. No scaling. No normalization. No PCA. No get_angles.

### Define trainable layer ###
def trainable_layer(layer_weights):
    # Trainable single-qubit gates
    for i in range(num_qubits):
        # each qubit gets two trainable gates
        # RY(theta)
        # RZ(theta)
        qml.RY(layer_weights[i, 0], wires=i)
        qml.RZ(layer_weights[i, 1], wires=i)
    

    # Entangling gates
    for i in range(num_qubits - 1):
        # qubits are connected using a chain of CNOTs:
        # q0 → q1 → q2 → q3 → q4 → q5 → q6 → q7
        qml.CNOT(wires=[i, i + 1])

    #uses 7 two qubit gates per layer 
    # with 4 layers
    #  4 × 7 = 28 two-qubit gates, competition limit is 80

## full quantum circuit ## 

@qml.qnode(dev, interface="autograd")
def circuit(weights, bias, x):
    for layer_id in range(num_layers):
        encode_data(x) # each layer gets the freshly encoded data?
        trainable_layer(weights[layer_id])

    # In-circuit bias instead of classical + bias
    qml.RY(bias, wires=0)

    #for qubit 0
    # PauliZ expectation near +1 → likely measured as 0
    #PauliZ expectation near -1 → likely measured as 1 why ?
    return qml.expval(qml.PauliZ(0))


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

    n0 = np.sum(y_batch == 0)
    n1 = np.sum(y_batch == 1)
    total = len(y_batch)

    w0 = total / (2 * n0 + eps)
    w1 = total / (2 * n1 + eps)

    sample_weights = np.where(y_batch == 1, w1, w0)

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

onp.random.seed(0)

weights = 0.01 * np.array(
    onp.random.randn(num_layers, num_qubits, 2),
    requires_grad=True
)

bias = np.array(0.0, requires_grad=True)

###Train the model ###
opt = qml.AdamOptimizer(stepsize=0.03)

batch_size = 64
num_epochs = 100

best_bal_acc = 0
best_weights = None
best_bias = None

rng = onp.random.default_rng(0)

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


##### TEST #####
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
        flat_params.append(float(best_weights[layer_id, qubit_id, 0]))
        flat_params.append(float(best_weights[layer_id, qubit_id, 1]))

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

#### Translate the circuit into QASM #### 

#Encode x1...x8 directly as RY angles on 8 qubits → apply 4 layers of trainable RY/RZ gates plus CNOT entanglement → measure q[0] → train theta_0...theta_64 using training data only → submit QASM plus weights.json.
# Each row has eight raw features x1...x8. I encode them directly as rotation angles using RY(x_i) on qubit i, with no scaling, normalization, feature engineering, PCA, or data augmentation. The same direct encoding is repeated in each data re-uploading layer.

def generate_classifier_qasm(
    filename="classifier.qasm",
    num_qubits=8,
    num_layers=4
):
    lines = []

    lines.append("OPENQASM 3.0;")
    lines.append('include "stdgates.inc";')
    lines.append("")

    # Data input placeholders
    # These represent x1...x8 from each row of the dataset.
    for i in range(num_qubits):
        lines.append(f"input float[64] x_{i};")

    lines.append("")

    # Trainable parameter placeholders
    # For 4 layers, 8 qubits, 2 trainable gates per qubit:
    # (num layers)(weights or params per layer) + final bias or weight = 65
    # 4 * 8 * 2 = 64
    # plus one final bias parameter = 65
    num_trainable_params = num_layers * num_qubits * 2 + 1

    for i in range(num_trainable_params):
        lines.append(f"input float[64] theta_{i};")

    lines.append("")
    lines.append(f"qubit[{num_qubits}] q;")
    lines.append("bit result;")
    lines.append("")

    theta_index = 0

    for layer in range(num_layers):
        lines.append(f"// Layer {layer + 1}: direct data encoding")

        # Direct data encoding
        for qubit in range(num_qubits):
            lines.append(f"ry(x_{qubit}) q[{qubit}];")

        lines.append(f"// Layer {layer + 1}: trainable rotations")

        # Trainable single-qubit gates
        for qubit in range(num_qubits):
            lines.append(f"ry(theta_{theta_index}) q[{qubit}];")
            theta_index += 1

            lines.append(f"rz(theta_{theta_index}) q[{qubit}];")
            theta_index += 1

        lines.append(f"// Layer {layer + 1}: entanglement")

        # CNOT chain
        for qubit in range(num_qubits - 1):
            lines.append(f"cx q[{qubit}], q[{qubit + 1}];")

        lines.append("")

    # Final trainable bias rotation on measured qubit q[0]
    lines.append("// Final trainable bias rotation")
    lines.append(f"ry(theta_{theta_index}) q[0];")
    lines.append("")

    # One-qubit measurement
    lines.append("// Measure one qubit")
    lines.append("result = measure q[0];")
    lines.append("")

    qasm_text = "\n".join(lines)

    with open(filename, "w") as f:
        f.write(qasm_text)

    print(f"Saved {filename}")
    print(f"Number of trainable parameters: {num_trainable_params}")
    print(f"Last theta used: theta_{theta_index}")


generate_classifier_qasm()