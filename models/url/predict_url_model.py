import joblib
import numpy as np
import pandas as pd
import re
import tldextract
from urllib.parse import urlparse, parse_qs
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and vectorizer
MODEL_PATH = 'models/url/url_model.pkl'
VECTORIZER_PATH = 'models/url/url_vectorizer.pkl'

model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

def extract_features(url):
    """
    Extract extensive set of URL features.
    """
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

def predict_url_spam(url):
    """
    Predict whether a given URL is spam or not using the trained model.
    """
    # Extract structured features
    structured_features = np.array(extract_features(url)).reshape(1, -1)

    # Generate TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([url])

    # Combine structured and TF-IDF features
    combined_features = hstack([structured_features, tfidf_features])

    # Make a prediction
    prediction = model.predict(combined_features)
    return int(prediction[0] > 0.5)  # Return 1 for spam, 0 for not spam

def evaluate_against_sms_url_combined(file_path):
    """
    Evaluate the URL spam model against the `sms_url_combined.csv` file.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    y_true = df['is_spam'].values
    urls = df['text'].astype(str).values

    # Predict for all URLs
    y_pred = []
    for url in urls:
        try:
            y_pred.append(predict_url_spam(url))
        except Exception as e:
            print(f"Error processing URL: {url}, Error: {e}")
            y_pred.append(0)  # Default to not spam in case of error

    # Calculate accuracy and classification report
    print(f"length of y_true: {len(y_true)}")
    print(f"length of y_pred: {len(y_pred)}")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    file_path = 'data/analysis/sms_url_combined.csv'  # Path to the dataset
    evaluate_against_sms_url_combined(file_path)