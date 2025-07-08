import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load ratio-only dataset
df = pd.read_csv("ratio.csv")
X = df.drop(columns=["labels"]).values.astype(np.float32)
y = df["labels"].values.astype(np.float32)

# Treat 6 ratio features as a pseudo-sequence of 6 timesteps with 1 feature each
X = X.reshape(X.shape[0], X.shape[1], 1)  # shape = (N, 6, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# GRU model
def gru_forward(x, is_training=True):
    gru = hk.GRU(8)
    core = hk.StaticUnroll(gru, length=x.shape[1])
    output, state = core(x, gru.initial_state(x.shape[0]))

    mlp = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Dropout(0.2),
        hk.Linear(1), jax.nn.sigmoid
    ])
    return mlp(state)

model = hk.transform(gru_forward)
rng = jax.random.PRNGKey(42)
params = model.init(rng, jnp.array(X_train[:4]), is_training=True)

optimizer = optax.adamw(1e-3, weight_decay=1e-4)
opt_state = optimizer.init(params)

# Loss function
def compute_loss(params, x, labels):
    preds = model.apply(params, rng, x, is_training=True).squeeze()
    return optax.sigmoid_binary_cross_entropy(preds, labels).mean()

@jax.jit
def update(params, opt_state, x, labels):
    loss, grads = jax.value_and_grad(compute_loss)(params, x, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
batch_size = 8
for epoch in range(1, 21):
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        params, opt_state, _ = update(params, opt_state, jnp.array(x_batch), jnp.array(y_batch))

    # Evaluation
    y_pred = []
    for i in range(len(X_test)):
        x_i = jnp.expand_dims(X_test[i], axis=0)
        out = model.apply(params, rng, x_i, is_training=False).squeeze()
        y_pred.append(float(out > 0.5))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Epoch {epoch:02d} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
