import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import optax
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('Text.csv')
X_text = df['text'].values
y = df['labels'].values.astype(np.float32)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(X_text).toarray().astype(np.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create batched graph
def create_graph_batch(X_batch):
    return jraph.GraphsTuple(
        nodes=jnp.array(X_batch),
        edges=None,
        senders=jnp.array([], dtype=jnp.int32),
        receivers=jnp.array([], dtype=jnp.int32),
        n_node=jnp.array([X_batch.shape[0]]),
        n_edge=jnp.array([0]),
        globals=jnp.array([0.0])
    )

# GNN model definition
def model_fn(graph):
    net = jraph.GraphNetwork(
        update_node_fn=lambda n, s, r, g: hk.Sequential([
            hk.Linear(16), jax.nn.relu,
            hk.Dropout(0.2),
            hk.Linear(8), jax.nn.relu
        ])(n),
        update_edge_fn=None,
        update_global_fn=lambda n, e, g: hk.Sequential([
            hk.Linear(4), jax.nn.relu,
            hk.Linear(1), jax.nn.sigmoid
        ])(g),
    )
    return net(graph).globals

# Transform + init
model = hk.transform(model_fn)
rng = jax.random.PRNGKey(42)
sample_graph = create_graph_batch(np.expand_dims(X_train[0], axis=0))
params = model.init(rng, sample_graph)

# Optimizer with weight decay
optimizer = optax.adamw(1e-3, weight_decay=1e-4)
opt_state = optimizer.init(params)

# Loss
def loss_fn(params, graph, y_true):
    pred = model.apply(params, rng, graph).squeeze()
    return optax.sigmoid_binary_cross_entropy(pred, y_true).mean()

@jax.jit
def update(params, opt_state, graph, y_true):
    grads = jax.grad(loss_fn)(params, graph, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state

# Training loop
for epoch in range(1, 21):
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(len(X_train_shuffled)):
        x_i = np.expand_dims(X_train_shuffled[i], axis=0)
        y_i = np.array([y_train_shuffled[i]])
        g = create_graph_batch(x_i)
        params, opt_state = update(params, opt_state, g, jnp.array(y_i))

    # Evaluate
    y_pred = []
    for i in range(len(X_test)):
        g = create_graph_batch(np.expand_dims(X_test[i], axis=0))
        out = model.apply(params, rng, g).squeeze()
        y_pred.append(out > 0.5)

    y_pred = np.array(y_pred).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Epoch {epoch:02d} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
