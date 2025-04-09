import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, accuracy_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
import matplotlib.pyplot as plt
import seaborn as sns

# Choose model to run: 'dt', 'knn', 'nb'
model_type = 'nb'

# Load Data
print("Loading data...")
sms_df = pd.read_csv('data/normalized/sms_uci_data.csv', keep_default_na=False)
print("Data loaded!")

# Optimized TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 7),
    analyzer='char_wb',
    max_features=40000,
    min_df=1,
    max_df=0.95,
    stop_words='english',
    sublinear_tf=True
)

X = vectorizer.fit_transform(sms_df['text'])
y = sms_df['is_spam'].values

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, aucs, avg_precisions = [], [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

print(f"\nStarting Stratified K-Fold training for {model_type.upper()}...\n")
start = time.time()
fold = 1
for train_idx, test_idx in skf.split(X, y):
    print(f"--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if model_type == 'dt':
        model = DecisionTreeClassifier(max_depth=50, min_samples_split=5, random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')
    elif model_type == 'nb':
        model = ComplementNB(alpha=0.001, fit_prior=True)
    else:
        raise ValueError("Invalid model_type selected")

    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    accuracies.append(acc)
    aucs.append(auc)
    avg_precisions.append(avg_prec)

    print(f"Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Avg Precision: {avg_prec:.4f}, Time: {t1 - t0:.2f}s\n")
    fold += 1

end = time.time()
total_time = end - start

# Final Metrics
final_cm = confusion_matrix(all_y_true, all_y_pred)
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_proba)
final_auc = roc_auc_score(all_y_true, all_y_proba)
final_avg_prec = average_precision_score(all_y_true, all_y_proba)
final_accuracy = accuracy_score(all_y_true, all_y_pred)

# Visualization
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# --- ROC Curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {final_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(f'analysis/sms/{model_type}/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {final_avg_prec:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.savefig(f'analysis/sms/{model_type}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f'analysis/sms/{model_type}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Final Report
print(f"\n{' FINAL EVALUATION (' + model_type.upper() + ') ':=^60}\n")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average AUC-ROC: {np.mean(aucs):.4f}")
print(f"Average Precision: {np.mean(avg_precisions):.4f}\n")
print("Classification Report (aggregated predictions):")
print(classification_report(all_y_true, all_y_pred, target_names=['Legitimate', 'Spam']))
