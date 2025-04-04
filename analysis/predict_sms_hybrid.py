import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re
from urllib.parse import urlparse, parse_qs
import tldextract   

# Load Pretrained SMS Model and Vectorizer
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load the trained URL spam detection model and vectorizer
url_model = joblib.load('models/url/url_model.pkl')
url_vectorizer = joblib.load('models/url/url_vectorizer.pkl')

# Load the combined SMS and URL dataset
data = pd.read_csv('sms_url_combined.csv', keep_default_na=False)

# Function to extract features for the URL model
def extract_url_features(url):
    """Extract extensive set of URL features."""
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)
    
    domain = domain_info.domain
    suffix = domain_info.suffix
    subdomain = domain_info.subdomain
    path = parsed.path
    query = parsed.query

    # Calculate entropy (higher entropy = more randomness, likely spam)
    def entropy(string):
        prob = [float(string.count(c)) / len(string) for c in set(string)]
        return -sum(p * np.log2(p) for p in prob)

    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'entropy_url': entropy(url),
        'num_subdomains': subdomain.count('.') + 1 if subdomain else 0,
        'domain_length': len(domain),
        'path_length': len(path),
        'query_length': len(query),
        'num_query_params': len(parse_qs(query)),
        'tld_length': len(suffix),
        'suspicious_tld': int(suffix in ['xyz', 'top', 'click', 'club', 'biz', 'info', 'work', 'zip', 'mobi']),
        'has_ip_address': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        'contains_free': int('free' in url.lower()),
        'contains_win': int(any(word in url.lower() for word in ['win', 'reward', 'gift', 'claim'])),
        'contains_login': int('login' in url.lower()),
        'contains_auth': int('auth' in url.lower()),
        'contains_account': int('account' in url.lower()),
        'contains_offer': int('offer' in url.lower()),
        'contains_secure': int('secure' in url.lower()),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_slashes': url.count('/'),
    }
    return list(features.values())

# Initialize predictions list
predictions = []

# Iterate through each row in the dataset
for _, row in data.iterrows():
    text = row['text']

    # Extract and normalize URLs from the text
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    urls = [url if url.startswith(('http://', 'https://')) else 'http://' + url for url in urls]

    # Remove URLs from the SMS text
    clean_text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Predict using the SMS model
    X_text = text_vectorizer.transform([clean_text])
    sms_prediction = sms_model.predict(X_text)[0]

    # Predict using the URL model (if URLs are present)
    url_prediction = 0
    if urls:
        # Extract features for each URL
        url_features = np.array([extract_url_features(url) for url in urls])
        url_tfidf_features = url_vectorizer.transform(urls).toarray()
        combined_url_features = np.hstack([url_features, url_tfidf_features])

        # Predict using the URL model
        url_prediction = int(any(url_model.predict(combined_url_features)))

    # Combine predictions: if either model predicts spam, mark as spam
    final_prediction = sms_prediction or url_prediction
    predictions.append(final_prediction)

# Compare predictions with the actual labels
accuracy = accuracy_score(data['is_spam'], predictions)
report = classification_report(data['is_spam'], predictions)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)