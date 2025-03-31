import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

print("Loading data...")

# Load sms data
sms_df = pd.read_csv('data/normalized/sms_data.csv', keep_default_na=False)

print("Data loaded!")

# Vectorize the 'text' column using TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # Unigrams and bigrams
    max_features=7000,    # Capture more features (was 5000)
    min_df=3,             # Ignore terms appearing in <3 documents
    max_df=0.9,           # Ignore overly common terms
    stop_words='english'  # Remove English stopwords
)

# Fit and transform the text data
X = tfidf.fit_transform(sms_df['text'])
y = sms_df['is_spam']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the XGBoost model with optimal hyperparameters
model = XGBClassifier(
    n_estimators=400,         # Increase for more complex relationships
    max_depth=6,              # Prevents overfitting
    learning_rate=0.05,       # Lower to balance higher estimators
    subsample=0.8,            # Use 80% of data for each tree to prevent overfitting
    colsample_bytree=0.8,     # Add column subsampling to further reduce overfitting
    random_state=42,          # Ensures reproducibility
    eval_metric='logloss',    # Logarithmic loss for binary classification
    early_stopping_rounds=10  # Stop if no improvement in 10 rounds
)


# Start time for training
print("\nStarting training...")
start = time.time()

# Train the model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# End time for training
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")

# Evaluate the model
y_pred = model.predict(X_test)

# Print accuracy
print(f'\nAccuracy: {model.score(X_test, y_test):.4f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(tfidf, 'models/sms/sms_text_vectorizer.pkl')
joblib.dump(model, 'models/sms/sms_model.pkl')

print("\nSMS Spam Model and Vectorizer saved!")