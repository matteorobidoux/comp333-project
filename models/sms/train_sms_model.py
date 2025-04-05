import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("Loading data...")

# Load SMS data
sms_df = pd.read_csv('data/normalized/sms_data.csv', keep_default_na=False)
print("Data loaded!")

# Vectorization with TF-IDF (Unigrams, Bigrams, and Char-level n-grams)
tfidf = TfidfVectorizer(
    ngram_range=(1, 6),    # Unigrams, Bigrams, and Trigrams
    analyzer='char_wb',    # Works well for spam detection
    max_features=10000,    # More features for better classification
    min_df=2,              # Ignore terms appearing in <2 documents
    max_df=0.95,           # Ignore extremely common terms
    stop_words='english'   # Remove stopwords
)

# Fit and transform the text data
X = tfidf.fit_transform(sms_df['text'])
y = sms_df['is_spam']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define and train the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=300,  # Number of trees
    max_depth=50,      # Maximum depth of each tree
    random_state=42,   # For reproducibility
    n_jobs=-1          # Use all cores
)

# Training
print("\nStarting training...")
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()

print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability scores for class 1 (spam)

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
            xticklabels=['Legitimate', 'Spam'], 
            yticklabels=['Legitimate', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.savefig('analysis/sms/sms_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print Metrics ---
print(f"\n{' MODEL EVALUATION ':=^60}\n")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))

# Save Model & Vectorizer
joblib.dump(tfidf, 'models/sms/sms_text_vectorizer.pkl')
joblib.dump(rf_model, 'models/sms/sms_model.pkl')

print("\nSMS Spam Model and Vectorizer saved!")
