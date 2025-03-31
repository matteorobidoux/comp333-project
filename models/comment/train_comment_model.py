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

# Load and preprocess comments data
comments_df = pd.read_csv('data/normalized/comments_data.csv')

# Fill missing values in 'text' and 'author' columns
comments_df['text'] = comments_df['text'].fillna('').astype(str)
comments_df['author'] = comments_df['author'].fillna('').astype(str)

# ============================
# Step 2: Feature Engineering
# ============================

# Vectorize the 'text' column using TF-IDF (unigrams and bigrams, max 5000 features)
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    max_features=5000,   # Limit features to 5000 for text
    stop_words='english'  # Remove English stopwords
)

# Vectorize the 'author' column using TF-IDF (unigrams and bigrams, max 1000 features)
author_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1000,   # Limit features to 1000 for author
    stop_words='english'
)

# Fit and transform text and author columns
X_text = text_vectorizer.fit_transform(comments_df['text'])
X_author = author_vectorizer.fit_transform(comments_df['author'])

# Combine text and author features using horizontal stacking
X = hstack([X_author, X_text])

# ============================
# Step 3: Model Training
# ============================

# Prepare target labels for training
y = comments_df['is_spam']

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define the Gradient Boosting model with optimal hyperparameters
gb_model = GradientBoostingClassifier(
    n_estimators=200,    # Number of boosting stages (trees)
    max_depth=6,         # Maximum depth of trees to prevent overfitting
    learning_rate=0.1,   # Step size at each iteration
    random_state=42      # Set random state for reproducibility
)

print("Starting training...")
start = time.time()
print("start time:", start)
# Train the model
gb_model.fit(X_train, y_train)
end = time.time()
print("Training completed.")
print("end time:", end)
print("Training time:", end - start)

# ============================
# Step 4: Model Evaluation
# ============================

# Predict on the test set
y_pred = gb_model.predict(X_test)

# Print classification report to evaluate model performance
print("Gradient Boosting Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Vectorizers
# ============================

# Save the text vectorizer, author vectorizer, and trained model for future use
joblib.dump(text_vectorizer, 'models/comment/comment_text_vectorizer.pkl')
joblib.dump(author_vectorizer, 'models/comment/comment_author_vectorizer.pkl')
joblib.dump(gb_model, 'models/comment/comment_model.pkl')

print("✅ Comment Spam Model and Vectorizers saved successfully!")