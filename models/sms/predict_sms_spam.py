import re
import joblib
import numpy as np
from urllib.parse import urlparse
import sys
import json
from scipy.special import expit
import tldextract
from urllib.parse import parse_qs
import re

# Load Pretrained Models and Vectorizers
# Load the trained SMS text vectorizer and model
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load the trained URL spam detection model and vectorizer
url_model = joblib.load('models/url/url_model.pkl')
url_vectorizer = joblib.load('models/url/url_vectorizer.pkl')

# Strip scheme for neutralization
def strip_scheme(url):
    return re.sub(r'^https?:\/\/', '', url.lower())

urls = raw_urls.apply(strip_scheme).values

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
    """Ensure that the URL starts with a valid scheme and add 'www.' if missing."""
    if not re.match(r'^(http://|https://)', url):  # Add protocol if missing
        url = 'http://' + url
    if not re.match(r'^(http://www\.|https://www\.)', url):  # Add 'www.' if missing
        url = url.replace('http://', 'http://www.').replace('https://', 'https://www.')
    return url


def analyze_text(text):
    """Analyze the SMS text and any URLs within it to detect spam."""
    
    # Extract and normalize URLs from the text
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    urls = [fix_url(url) for url in urls]

    # Default probabilities for SMS and URL spam detection
    sms_spam_prob = 0.0
    url_spam_prob = 0.0

    # Analyze the SMS content after removing URLs
    clean_text = re.sub(r'https?://\S+|www\.\S+', '', text)

    X_text = text_vectorizer.transform([clean_text])

    # Get spam probability for the SMS content
    sms_spam_prob = sms_model.predict_proba(X_text)[0][1]

    # Analyze URL spam if any URLs are found
    if urls:
        # Extract features from each URL
        url_features = np.array([extract_url_features(url) for url in urls])
        
        # Apply TF-IDF vectorization on the URL features
        url_tfidf_features = url_vectorizer.transform([url for url in urls])
        
        # Combine the structured features with the TF-IDF features
        combined_url_features = np.hstack([url_features, url_tfidf_features.toarray()])
        
        # For LightGBM model, use `predict()` and apply sigmoid to get probabilities
        url_spam_prob = url_model.predict(combined_url_features)[0]
        # url_spam_prob = max(expit(raw_probs))  # Apply sigmoid to get probabilities



        # Combine SMS and URL spam probabilities (weighted sum)
        combined_prob = 0.3 * sms_spam_prob + 0.7 * url_spam_prob
    else:
        # If no URLs are found, use only the SMS spam probability
        combined_prob = sms_spam_prob

    # Determine if the text is spam based on the combined probability
    is_spam = combined_prob >= 0.5

    # Return the results in a dictionary
    return {
        'spam_prob': round(sms_spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }


if __name__ == '__main__':
    # Get the text input from the command line or use a default example
    param = sys.argv[1] if len(sys.argv) > 1 else 'Check out this link: http://example.com/free-gift'

    # Analyze the text input for spam
    prediction_result = analyze_text(param)

    # Convert any NumPy float values to standard Python floats for JSON serialization
    for key, value in prediction_result.items():
        if isinstance(value, np.float64) or isinstance(value, np.float32):
            prediction_result[key] = round(float(value), 2)

    # Print the result as a formatted JSON string
    print(json.dumps(prediction_result, indent=4))