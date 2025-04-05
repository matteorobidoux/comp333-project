import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load Pretrained SMS Model and Vectorizer
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load the combined SMS and URL dataset
data = pd.read_csv('data/analysis/sms_url_combined.csv', keep_default_na=False)

# Preprocess the text column using the vectorizer
X = text_vectorizer.transform(data['text'])

# Predict using the SMS model
predictions = sms_model.predict(X)

print(f"length of predictions: {len(predictions)}")
print(f"length of data['is_spam']: {len(data['is_spam'])}")

# Compare predictions with the actual labels
accuracy = accuracy_score(data['is_spam'], predictions)
report = classification_report(data['is_spam'], predictions)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)