import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# ============================
# Step 1: Load and Preprocess Data
# ============================

# Load and preprocess SMS data
sms_df = pd.read_csv('data/normalized/sms_data.csv')

# Check for missing or empty texts
sms_df['text'] = sms_df['text'].fillna('').astype(str)

# ============================
# Step 2: Feature Engineering
# ============================
# TF-IDF Vectorization
text_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # Unigrams and bigrams
    max_features=5000,     # Limit to top 5000 features
    stop_words='english'   # Remove common stopwords
)

X_text = text_vectorizer.fit_transform(sms_df['text'])

# ============================
# Step 3: Model Training
# ============================
# Prepare target labels
y = sms_df['is_spam']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y
)

# Define and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,     # Number of boosting stages (trees)
    max_depth=6,           # Maximum depth of trees
    learning_rate=0.1,     # Step size at each iteration
    random_state=42        # Random state for reproducibility
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
# Predict and evaluate performance
y_pred = gb_model.predict(X_test)
print("✅ Gradient Boosting Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Vectorizer
# ============================
# Save the trained model and vectorizer
joblib.dump(text_vectorizer, 'models/sms/sms_text_vectorizer.pkl')
joblib.dump(gb_model, 'models/sms/sms_model.pkl')

print("📚 SMS Model and Vectorizer saved successfully!")