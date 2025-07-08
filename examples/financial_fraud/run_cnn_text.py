import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load and clean data
df = pd.read_csv("Text.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["statements"])
df = df[df["statements"].astype(str).str.strip() != ""]
df = df.reset_index(drop=True)

# Extract inputs and labels
texts = df["statements"].astype(str).tolist()
labels = df["labels"].astype(int).values

# Tokenize text
max_words = 5000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=max_words, output_dim=128),
    Conv1D(filters=64, kernel_size=5, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=1)

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Final Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
