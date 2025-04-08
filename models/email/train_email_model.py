import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")

# Load email data
email_df = pd.read_csv('data/normalized/email_data.csv', keep_default_na=False)

print("Data loaded!")

# TF-IDF Vectorization for both subject and text
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # Unigrams and bigrams
    max_features=5000,     # Limit to top 5000 features
    min_df=3,              # Ignore terms appearing in fewer than 3 docs
    max_df=0.9,            # Ignore overly common terms
    stop_words='english'   # Remove common stopwords
)
subject_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # Unigrams and bigrams
    max_features=1000,    # Smaller limit for subject
    max_df=0.95,          # Ignore overly common terms
    min_df=2,             # Ignore rare terms
    stop_words='english'  # Remove common stopwords
)

# Fit and transform text and subject features
X_text = text_vectorizer.fit_transform(email_df['text'])
X_subject = subject_vectorizer.fit_transform(email_df['subject'])

# Combine text and subject features into a single matrix
X = hstack([X_subject, X_text])

# Prepare target labels
y = email_df['is_spam']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define and train the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=500,               # Number of boosting stages
    max_depth=10,                   # Maximum tree depth
    learning_rate=0.05,             # Step size at each iteration
    subsample=0.8,                  # Fraction of samples to use for each tree
    colsample_bytree=0.8,           # Fraction of features to consider at each split
    objective='binary:logistic',    # For binary classification
    eval_metric='logloss',          # Evaluation metric
    random_state=42,                # Ensures reproducibility
    early_stopping_rounds=30,       # Stop if no improvement in 30 rounds
)

# Start time training the model
print("\nStarting training...")
start = time.time()

# Train the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# End time for training
end = time.time()
total_time = end - start
print(f"Training completed in {total_time:.2f} seconds.")

# Predict and evaluate performance
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# --- Evaluation Metrics ---
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Generate curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# --- ROC Curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('analysis/email/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {avg_precision:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.savefig('analysis/email/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('analysis/email/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Print Metrics ---
print(f"\n{' MODEL EVALUATION ':=^60}\n")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save the trained model and vectorizers
joblib.dump(text_vectorizer, 'models/email/email_text_vectorizer.pkl')
joblib.dump(subject_vectorizer, 'models/email/email_subject_vectorizer.pkl')
joblib.dump(xgb_model, 'models/email/email_model.pkl')

print("\nEmail Spam Model and Vectorizers saved!")
