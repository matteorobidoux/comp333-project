import pandas as pd
import joblib
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")

# Load SMS data
sms_df = pd.read_csv('data/normalized/sms_uci_data.csv', keep_default_na=False)
print("Data loaded!")

# Vectorization with TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 6),
    analyzer='char_wb',
    max_features=10000,
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X = tfidf.fit_transform(sms_df['text'])
y = sms_df['is_spam'].values

# Stratified K-Fold Setup
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Metrics collectors
accuracies, aucs, avg_precisions = [], [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

print("\nStarting Stratified K-Fold training...\n")
fold = 1
for train_idx, test_idx in skf.split(X, y):
    print(f"--- Fold {fold} ---")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=50,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    rf_model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    # Store for final evaluation
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    accuracies.append(acc)
    aucs.append(auc)
    avg_precisions.append(avg_prec)

    print(f"Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Avg Precision: {avg_prec:.4f}, Time: {end_time - start_time:.2f}s\n")
    fold += 1

# Final evaluation
final_cm = confusion_matrix(all_y_true, all_y_pred)
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_proba)
final_auc = roc_auc_score(all_y_true, all_y_proba)
final_avg_prec = average_precision_score(all_y_true, all_y_proba)
final_accuracy = accuracy_score(all_y_true, all_y_pred)

# Visualization
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {final_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {final_avg_prec:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

# Confusion Matrix
plt.subplot(1, 3, 3)
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.savefig('analysis/sms/sms_model_performance_kfold.png', dpi=300, bbox_inches='tight')
plt.show()

# Print Final Metrics
print(f"\n{' FINAL EVALUATION (Stratified K-Fold) ':=^60}\n")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average AUC-ROC: {np.mean(aucs):.4f}")
print(f"Average Precision: {np.mean(avg_precisions):.4f}\n")
print("Classification Report (aggregated predictions):")
print(classification_report(all_y_true, all_y_pred, target_names=['Legitimate', 'Spam']))

# Save the last trained model and vectorizer
joblib.dump(tfidf, 'analysis/models/sms_text_vectorizer.pkl')
joblib.dump(rf_model, 'analysis/models/sms_model.pkl')

print("\nFinal SMS Spam Model and Vectorizer saved!")