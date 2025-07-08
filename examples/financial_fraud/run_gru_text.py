import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input

# Load and clean data
df = pd.read_csv('Text.csv', encoding='ISO-8859-1')
df = df.dropna(subset=['statements'])
df = df[df['statements'].astype(str).str.strip().ne('')]

# Balance dataset
df1 = df[df['labels'] == 1]
df0 = df[df['labels'] == 0].sample(n=len(df1), random_state=42)
df = pd.concat([df1, df0]).reset_index(drop=True)

# Text vectorization
vectorizer = TfidfVectorizer(max_features=200)
X = vectorizer.fit_transform(df['statements']).toarray()
y = df['labels'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for GRU
X_train_r = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_r = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build model
model = Sequential([
    Input(shape=(X.shape[1], 1)),
    GRU(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
for epoch in range(1, 21):
    model.fit(X_train_r, y_train, epochs=1, verbose=0)
    y_pred = (model.predict(X_test_r) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Epoch {epoch}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
