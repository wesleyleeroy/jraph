import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load ratio-only dataset
df = pd.read_csv("ratio.csv")
X = df.drop(columns=["labels"]).values.astype(np.float32)
y = df["labels"].values.astype(np.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build batched graph
def build_graph_batch(x_batch):
    return jraph.GraphsTuple(
        nodes=jnp.array(x_batch),
        edges=None,
        senders=jnp.array([], dtype=jnp.int32),
        receivers=jnp.array([], dtype=jnp.int32),
        globals=None,
        n_node=jnp.array([x_batch.shape[0]]),
        n_edge=jnp.array([0])
    )

# Define model
def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    mlp = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(1), jax.nn.sigmoid
    ])
    return graph._replace(nodes=mlp(graph.nodes))

# Transform with Haiku
net = hk.without_apply_rng(hk.transform(net_fn))
rng = jax.random.PRNGKey(42)
sample_graph = build_graph_batch(X_train[:4])
params = net.init(rng, sample_graph)

# Optimizer
optimizer = optax.adamw(1e-3, weight_decay=1e-4)
opt_state = optimizer.init(params)

# Loss function
def compute_loss(params, graph, labels):
    pred_graph = net.apply(params, graph)
    preds = pred_graph.nodes.squeeze()
    return optax.sigmoid_binary_cross_entropy(preds, labels).mean()

@jax.jit
def update(params, opt_state, graph, labels):
    loss, grads = jax.value_and_grad(compute_loss)(params, graph, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
batch_size = 8
for epoch in range(1, 21):
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]
        graph = build_graph_batch(x_batch)
        label_batch = jnp.array(y_batch)
        params, opt_state, _ = update(params, opt_state, graph, label_batch)

    # Evaluation
    y_pred = []
    for i in range(len(X_test)):
        graph = build_graph_batch(np.expand_dims(X_test[i], axis=0))
        pred = net.apply(params, graph).nodes[0]
        y_pred.append(float((pred > 0.5).item()))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Epoch {epoch:02d} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
