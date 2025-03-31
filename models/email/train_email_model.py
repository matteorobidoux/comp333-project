import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from xgboost import XGBClassifier

# ============================
# Step 1: Load and Preprocess Data
# ============================

# Load and preprocess email data
email_df = pd.read_csv('data/normalized/email_data.csv')

email_df['text'] = email_df['text'].fillna('').astype(str)
email_df['subject'] = email_df['subject'].fillna('').astype(str)

# ============================
# Step 2: Feature Engineering
# ============================
# TF-IDF Vectorization for both subject and text
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # Unigrams and bigrams
    max_features=5000,     # Limit to top 5000 features
    stop_words='english'   # Remove common stopwords
)
subject_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1000,     # Smaller limit for subject
    stop_words='english'
)

# Fit and transform text and subject features
X_text = text_vectorizer.fit_transform(email_df['text'])
X_subject = subject_vectorizer.fit_transform(email_df['subject'])

# Combine text and subject features into a single matrix
X = hstack([X_subject, X_text])

# ============================
# Step 3: Model Training
# ============================
# Prepare target labels
y = email_df['is_spam']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define and train the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=500,         # Number of boosting stages
    max_depth=10,             # Maximum tree depth
    learning_rate=0.05,       # Step size at each iteration
    subsample=0.8,            # Fraction of samples to use for each tree
    colsample_bytree=0.8,     # Fraction of features to consider at each split
    objective='binary:logistic',  # For binary classification
    eval_metric='logloss',
    random_state=42           # Ensures reproducibility
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# ============================
# Step 4: Model Evaluation
# ============================
# Predict and evaluate performance
y_pred = xgb_model.predict(X_test)
print("XGBoost Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Vectorizers
# ============================
# Save the trained model and vectorizers
joblib.dump(text_vectorizer, 'models/email/email_text_vectorizer.pkl')
joblib.dump(subject_vectorizer, 'models/email/email_subject_vectorizer.pkl')
joblib.dump(xgb_model, 'models/email/email_model.pkl')

print("Email Spam Model and Vectorizers saved successfully!")
