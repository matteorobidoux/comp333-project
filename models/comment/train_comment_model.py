import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time

print("Loading data...")

# Load comments data
comments_df = pd.read_csv('data/normalized/comment_data.csv', keep_default_na=False)

print("\nData loaded!")

# Vectorize the 'text' column using TF-IDF
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=5000,   # Limit to top 5000 features
    min_df=3,            # Ignore terms appearing in fewer than 3 docs
    max_df=0.9,          # Ignore overly common terms
    stop_words='english' # Remove English stopwords
)

# Vectorize the 'author' column using TF-IDF
author_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=1000,   # Limit to top 1000 features
    min_df=2,            # Ignore rare terms
    max_df=0.95,         # Ignore overly common terms
    stop_words='english' # Remove English stopwords
)

# Fit and transform text and author columns
X_text = text_vectorizer.fit_transform(comments_df['text'])
X_author = author_vectorizer.fit_transform(comments_df['author'])

# Combine text and author features using horizontal stacking
X = hstack([X_author, X_text])

# Prepare target labels for training
y = comments_df['is_spam']

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y 
)

# Define the Gradient Boosting model with optimal hyperparameters
gb_model = GradientBoostingClassifier(
    n_estimators=150,        # Slightly reduced for smaller data
    max_depth=5,             # Shallow depth to avoid overfitting
    learning_rate=0.1,       # Step size for training
    random_state=42,         # Ensures reproducibility
    subsample=0.8,           # Use a fraction of data for each tree
    min_samples_split=2,     # Minimum samples required to split nodes
    min_samples_leaf=1,      # Minimum samples required to be a leaf node
)

# Start time for training the model
start = time.time()
print("\nStarting training...")

# Train the model
gb_model.fit(X_train, y_train)

# End time for training the model
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")

# Predict on the test set
y_pred = gb_model.predict(X_test)

# Print Accuracy score
print(f'\nAccuracy: {gb_model.score(X_test, y_test):.4f}')

# Print classification report to evaluate model performance
print("\nGradient Boosting Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save the text vectorizer, author vectorizer, and trained model for future use
joblib.dump(text_vectorizer, 'models/comment/comment_text_vectorizer.pkl')
joblib.dump(author_vectorizer, 'models/comment/comment_author_vectorizer.pkl')
joblib.dump(gb_model, 'models/comment/comment_model.pkl')

print("\nComment Spam Model and Vectorizers saved!")