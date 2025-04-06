import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack
from scipy.special import expit
import sys
import json
import tldextract
from urllib.parse import parse_qs
import re

# Load Pretrained Models and Vectorizers
# Load vectorizers and models for email spam classification
text_vectorizer = joblib.load('models/email/email_text_vectorizer.pkl')
subject_vectorizer = joblib.load('models/email/email_subject_vectorizer.pkl')
email_model = joblib.load('models/email/email_model.pkl')

# Load the URL feature extraction model
url_model = joblib.load('models/url/url_model.pkl')
url_vectorizer = joblib.load('models/url/url_vectorizer.pkl')  # New vectorizer for URL feature extraction

# Strip scheme for neutralization
def strip_scheme(url):
    return re.sub(r'^https?:\/\/', '', url.lower())

# --- Feature Engineering ---
def extract_features(url):
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

def fix_url(url):
    """Ensure URLs have the correct protocol and format."""
    url = strip_scheme(url)
    if not re.match(r'^(http://|https://)', url):  # Add protocol if missing
        url = 'http://' + url
    if not re.match(r'^(http://www\.|https://www\.)', url):  # Add 'www.' if missing
        url = url.replace('http://', 'http://www.').replace('https://', 'https://www.')
    return url

def analyze_email(subject, text):
    """Analyze an email subject and text to detect spam and check URLs."""
    
    # Extract URLs from the subject and text
    urls = re.findall(r'https?://\S+|www\.\S+', subject + " " + text)
    urls = [fix_url(url) for url in urls]  # Normalize the URLs

    # Default spam probability for URLs (if none found)
    url_spam_prob = 0.0

    # Clean the subject and text by removing URLs
    clean_subject = re.sub(r'https?://\S+|www\.\S+', '', subject)
    clean_text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Vectorize the cleaned subject and text using the trained vectorizers
    X_subject = subject_vectorizer.transform([clean_subject])
    X_text = text_vectorizer.transform([clean_text])

    # Combine the subject and text vectors
    X_combined = hstack([X_subject, X_text])

    # Predict spam probability using the email model
    spam_prob = email_model.predict_proba(X_combined)[0][1]

    combined_prob = 0.0  # Initialize combined spam probability

    # Analyze URL spam if any URLs are present
    if urls:
        # Extract features from each URL
        url_features = np.array([extract_url_features(url) for url in urls])
         # Apply TF-IDF vectorization on the URL features
        url_tfidf_features = url_vectorizer.transform([url for url in urls])
        
        # Combine the structured features with the TF-IDF features
        combined_url_features = np.hstack([url_features, url_tfidf_features.toarray()])
        
        # For LightGBM model, use `predict()` and apply sigmoid to get probabilities
        raw_probs = url_model.predict(combined_url_features)
        url_spam_prob = max(expit(raw_probs))  # Apply sigmoid to get probabilities
        # Combine email and URL spam probabilities
        combined_prob = 0.3 * spam_prob + 0.7 * url_spam_prob
    else:
        combined_prob = spam_prob

    # Determine if the email is spam based on the combined probability
    is_spam = combined_prob >= 0.5

    # Return the analysis results
    return {
        'spam_prob': round(spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }

if __name__ == '__main__':
    # Get email subject and text from command line arguments
    subject = sys.argv[1] if len(sys.argv) > 1 else 'Test'
    text = sys.argv[2] if len(sys.argv) > 2 else 'Good afternoon, when will the test be? Thank you'

    # Run the spam analysis
    prediction_result = analyze_email(subject, text)

    # Convert NumPy float values to native Python floats for JSON serialization
    for key, value in prediction_result.items():
        if isinstance(value, np.float64) or isinstance(value, np.float32):
            prediction_result[key] = round(float(value), 2)

    # Print the result as a formatted JSON string
    print(json.dumps(prediction_result, indent=4))
