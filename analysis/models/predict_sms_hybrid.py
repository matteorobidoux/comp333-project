import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from urllib.parse import urlparse, parse_qs
import tldextract
import re

# Load Pretrained SMS Model and Vectorizer
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load Pretrained URL Model and Vectorizer
url_model = joblib.load('models/url/url_model.pkl')
url_vectorizer = joblib.load('models/url/url_vectorizer.pkl')

def strip_scheme(url):
    return re.sub(r'^https?:\/\/', '', url.lower())

def extract_url_features(url):
    parsed = urlparse('http://' + url)  # Add dummy scheme
    domain_info = tldextract.extract(url)

    domain = domain_info.domain
    suffix = domain_info.suffix
    subdomain = domain_info.subdomain
    path = parsed.path
    query = parsed.query

    def entropy(s):
        probs = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in probs)

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
        'contains_free': int('free' in url),
        'contains_win': int(any(word in url for word in ['win', 'reward', 'gift', 'claim'])),
        'contains_login': int('login' in url),
        'contains_auth': int('auth' in url),
        'contains_account': int('account' in url),
        'contains_offer': int('offer' in url),
        'contains_secure': int('secure' in url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_slashes': url.count('/'),
    }
    return list(features.values())

def predict_url_spam(url):
    """
    Predict whether a given URL is spam or not using the trained URL model.
    """
    # Normalize URL
    url = strip_scheme(url)

    # Extract structured features
    structured_features = np.array(extract_url_features(url)).reshape(1, -1)

    # Generate TF-IDF features
    tfidf_features = url_vectorizer.transform([url])

    # Combine structured and TF-IDF features
    combined_features = hstack([structured_features, tfidf_features])

    # Make a prediction
    prediction = url_model.predict(combined_features)
    return int(prediction[0] > 0.5)  # Return 1 for spam, 0 for not spam

def extract_url_from_text(text):
    """
    Extract the first URL from the text and return the URL and the text with the URL removed.
    """
    url_pattern = r'https?://\S+|www\.\S+'
    match = re.search(url_pattern, text)
    if match:
        url = match.group(0)
        text_without_url = re.sub(url_pattern, '', text).strip()
        return url, text_without_url
    return None, text  # No URL found, return the original text

def evaluate_hybrid_model(file_path):
    """
    Evaluate the hybrid SMS and URL spam detection model.
    """
    # Load the dataset
    data = pd.read_csv(file_path, keep_default_na=False)
    y_true = data['is_spam'].values
    texts = data['text'].astype(str).values

    sms_predictions = []
    url_predictions = []

    for text in texts:
        # Extract URL and clean SMS text
        url, clean_text = extract_url_from_text(text)

        # Predict using the SMS model
        X_sms = text_vectorizer.transform([clean_text])
        sms_pred = sms_model.predict(X_sms)[0]
        sms_predictions.append(sms_pred)

        # Predict using the URL model if a URL is found
        if url:
            try:
                url_pred = predict_url_spam(url)
            except Exception as e:
                print(f"Error processing URL: {url}, Error: {e}")
                url_pred = 0  # Default to not spam in case of error
        else:
            url_pred = 0  # No URL found, default to not spam
        url_predictions.append(url_pred)

    # Combine predictions: 1 if either SMS or URL predicts spam, 0 otherwise
    combined_predictions = [
        1 if sms == 1 or url == 1 else 0
        for sms, url in zip(sms_predictions, url_predictions)
    ]

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_true, combined_predictions)
    report = classification_report(y_true, combined_predictions)

    # Print results
    print(f"\nHybrid Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    # Path to the `sms_url_combined.csv` file
    file_path = 'data/analysis/sms_url_combined.csv'
    evaluate_hybrid_model(file_path)