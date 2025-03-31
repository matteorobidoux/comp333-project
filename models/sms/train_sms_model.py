import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import classification_report
from xgboost import XGBClassifier  # Better than GradientBoosting!

# ============================
# 1. Load & Preprocess Data
# ============================
sms_df = pd.read_csv('data/normalized/sms_data.csv')
sms_df['text'] = sms_df['text'].fillna('').astype(str)

# ============================
# 2. TF-IDF Vectorization (Keep it Simple)
# ============================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),  # Bigrams help catch phrases like "claim now"
    max_features=5000,   # Balances speed & accuracy
    stop_words='english'
)
X = tfidf.fit_transform(sms_df['text'])
y = sms_df['is_spam']

# ============================
# 3. Train/Test Split (80/20)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 4. Train XGBoost (Best Balance of Speed & Accuracy)
# ============================
model = XGBClassifier(
    n_estimators=300,      # More trees = better generalization (but slower)
    max_depth=6,           # Prevents overfitting
    learning_rate=0.1,     # Default works well
    subsample=0.8,         # Randomly samples data to reduce overfitting
    random_state=42,
    eval_metric='logloss'  # Better for binary classification
)


print("Starting training...")
start = time.time()
print("start time:", start)
# Train the model
model.fit(X_train, y_train)
end = time.time()
print("Training completed.")
print("end time:", end)
print("Training time:", end - start)


# ============================
# 5. Evaluate
# ============================
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ============================
# 6. Save Model
# ============================
joblib.dump(tfidf, 'models/sms/sms_text_vectorizer.pkl')
joblib.dump(model, 'models/sms/sms_model.pkl')
print("✅ Model saved!")