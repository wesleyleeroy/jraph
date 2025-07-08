import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# === RATIO DATA LOADER ===
def load_ratio_data(model_type='gnn'):
    df = pd.read_csv('ratio.csv')
    X = df.drop('labels', axis=1).values.astype(np.float32)
    y = df['labels'].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For GRU: reshape to [batch, time_steps=1, features]
    if model_type == 'gru':
        X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)

# === TEXT DATA LOADER ===
def load_text_data(model_type='gnn'):
    df = pd.read_csv('Text.csv')
    X_text = df['text'].astype(str).fillna('').values
    y = df['labels'].values.astype(np.float32)

    vectorizer = TfidfVectorizer(max_features=200)
    X = vectorizer.fit_transform(X_text).toarray().astype(np.float32)

    # For GRU: reshape to [batch, time_steps=1, features]
    if model_type == 'gru':
        X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)
