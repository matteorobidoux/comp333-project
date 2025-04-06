import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# Load model and vectorizer
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load and transform data
data = pd.read_csv('data/analysis/sms_url_combined.csv', keep_default_na=False)
X = text_vectorizer.transform(data['text'])
y_true = data['is_spam']

# Predict
y_pred = sms_model.predict(X)
y_proba = sms_model.predict_proba(X)[:, 1]

# Metrics
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
final_cm = confusion_matrix(y_true, y_pred)
final_auc = roc_auc_score(y_true, y_proba)
final_avg_prec = average_precision_score(y_true, y_proba)
fpr, tpr, _ = roc_curve(y_true, y_proba)
precision, recall, _ = precision_recall_curve(y_true, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

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
plt.text(0.6, 0.2, 
         f'Accuracy = {accuracy:.3f}', 
         fontsize=12,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

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
plt.savefig('analysis/sms/sms_traditional_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
