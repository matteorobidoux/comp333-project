import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, accuracy_score,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Load and preprocess data ===
df = pd.read_csv('data/normalized/sms_uci_data.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)
texts = df['text'].tolist()
labels = df['is_spam'].values

# === Tokenization ===
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = min(max(len(x) for x in sequences), 50)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = labels

# === K-Fold Setup ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y_true, all_y_pred, all_y_proba = [], [], []
accuracies, aucs, avg_precisions = [], [], []

# === Training Loop ===
for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Sequential([
        Embedding(input_dim=2000, output_dim=32, input_length=max_len),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(16)),
        Dropout(0.4),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
    ]

    model.fit(X_train, y_train, epochs=30, batch_size=32,
              validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

    accuracies.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_proba))
    avg_precisions.append(average_precision_score(y_test, y_proba))

# === Final Metrics ===
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

final_cm = confusion_matrix(all_y_true, all_y_pred)
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_proba)
final_auc = roc_auc_score(all_y_true, all_y_proba)
final_avg_prec = average_precision_score(all_y_true, all_y_proba)
final_accuracy = accuracy_score(all_y_true, all_y_pred)

# === Save Visuals ===
model_type = "LSTM"
output_dir = f'analysis/sms/{model_type}'
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {final_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {final_avg_prec:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# === Final Report ===
print(f"\n{' FINAL EVALUATION (LSTM BENCHMARK) ':=^60}")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average AUC-ROC: {np.mean(aucs):.4f}")
print(f"Average Precision: {np.mean(avg_precisions):.4f}")
print("\nClassification Report (aggregated predictions):")
print(classification_report(all_y_true, all_y_pred, target_names=['Legitimate', 'Spam']))
