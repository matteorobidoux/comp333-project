import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
from xgboost import XGBClassifier

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
print(f"Training completed in {end - start:.2f} seconds.")

# Predict and evaluate performance
y_pred = xgb_model.predict(X_test)

# Print Accuracy
print(f'\nAccuracy: {xgb_model.score(X_test, y_test):.4f}')

# Print classification report for detailed evaluation
print("\nXGBoost Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizers
joblib.dump(text_vectorizer, 'models/email/email_text_vectorizer.pkl')
joblib.dump(subject_vectorizer, 'models/email/email_subject_vectorizer.pkl')
joblib.dump(xgb_model, 'models/email/email_model.pkl')

print("\nEmail Spam Model and Vectorizers saved!")
