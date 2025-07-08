import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load and prepare dataset
df = pd.read_csv("ratio.csv")
X = df.drop(columns=["labels"]).values.astype(np.float32)
y = df["labels"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train[:, None, :]  # shape (N, 1, F)
X_test = X_test[:, None, :]

# CNN model with dropout and wider filters
def cnn_model(x, is_training=True):
    return hk.Sequential([
        hk.Conv1D(output_channels=16, kernel_shape=3, stride=1, padding='VALID'),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        hk.Dropout(0.2),
        hk.Linear(1), jax.nn.sigmoid
    ])(x)

def forward(x, is_training=True):
    return cnn_model(x, is_training)

model = hk.transform(forward)
rng = jax.random.PRNGKey(42)
params = model.init(rng, jnp.array(X_train[:4]), is_training=True)

optimizer = optax.adamw(1e-3, weight_decay=1e-4)
opt_state = optimizer.init(params)

# Loss function
def compute_loss(params, x, labels):
    preds = model.apply(params, rng, x, is_training=True).squeeze()
    return optax.sigmoid_binary_cross_entropy(preds, labels).mean()

@jax.jit
def update(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(compute_loss)(params, x, y)
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
