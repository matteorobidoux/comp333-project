import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
joblib.dump(tfidf, 'models/sms/sms_text_vectorizer.pkl')
joblib.dump(rf_model, 'models/sms/sms_model.pkl')

print("\nSMS Spam Model and Vectorizer saved!")
