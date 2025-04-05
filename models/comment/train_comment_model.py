import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score


print("Loading and preparing data...")

# Load data with more careful NA handling
comments_df = pd.read_csv('data/normalized/comment_data.csv', keep_default_na=False)

print("Data loaded!")

# Enhanced text preprocessing pipeline
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),           # Try trigrams
    max_features=8000,            # Increased features
    min_df=2,                     # Slightly more aggressive
    max_df=0.85,                  # Filter more common terms
    stop_words='english',
    sublinear_tf=True,            # Use logarithmic TF scaling
    analyzer='char_wb',           # Try character n-grams
    strip_accents='unicode',
    lowercase=True
)

# Author features with more aggressive settings
author_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1500,
    min_df=1,
    max_df=0.9,
    stop_words='english'
)

# Vectorize text and author features
X_text = text_vectorizer.fit_transform(comments_df['text'])
X_author = author_vectorizer.fit_transform(comments_df['author'])
X = hstack([X_author, X_text])
y = comments_df['is_spam']


# Stratified split with more test data for better evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Optimized Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,            # Increased number of trees
    max_depth=6,                 # Slightly deeper
    learning_rate=0.05,          # Smaller learning rate
    random_state=42,
    subsample=0.7,               # More aggressive subsampling
    min_samples_split=5,         # Require more samples to split
    min_samples_leaf=2,          # Require more samples per leaf
    max_features='sqrt',         # Consider sqrt of features per split
    validation_fraction=0.1,     # Early stopping
    n_iter_no_change=10
)

print("\nTraining model...")
start = time.time()
gb_model.fit(X_train, y_train)
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")

# Evaluation
y_pred = gb_model.predict(X_test)
y_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# --- Evaluation Metrics ---
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Generate curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# --- Visualization ---
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# 1. ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# 2. Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {avg_precision:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

# 3. Confusion Matrix
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.savefig('analysis/comment/comment_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print Metrics ---
print(f"\n{' MODEL EVALUATION ':=^60}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Save components
joblib.dump(text_vectorizer, 'models/comment/comment_text_vectorizer.pkl')
joblib.dump(author_vectorizer, 'models/comment/comment_author_vectorizer.pkl')
joblib.dump(gb_model, 'models/comment/comment_model.pkl')

print("\nModel saved!")