import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time

# ============================
# Step 1: Load and Preprocess Data
# ============================

# Load and preprocess email data
email_df = pd.read_csv('data/normalized/email_data.csv')

# Fill missing values in 'text' and 'subject' columns
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

X_text = text_vectorizer.fit_transform(email_df['text'])
X_subject = subject_vectorizer.fit_transform(email_df['subject'])

# Combine text and subject features
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

# Define and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,     # Number of boosting stages (trees)
    max_depth=6,           # Maximum depth of trees
    learning_rate=0.1,     # Step size at each iteration
    random_state=42        # Random state for reproducibility
)

# Train the model
print("Starting training...")
start = time.time()
print("start time:", start)
gb_model.fit(X_train, y_train)
end = time.time()
print("Training completed.")
print("end time:", end)
print("Training time:", end - start)

# ============================
# Step 4: Model Evaluation
# ============================
# Predict and evaluate performance
y_pred = gb_model.predict(X_test)
print("✅ Gradient Boosting Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Vectorizers
# ============================
# Save the trained model and vectorizers
joblib.dump(text_vectorizer, 'models/email/email_text_vectorizer.pkl')
joblib.dump(subject_vectorizer, 'models/email/email_subject_vectorizer.pkl')
joblib.dump(gb_model, 'models/email/email_model.pkl')

print("📚 Email Spam Model and Vectorizers saved successfully!")
