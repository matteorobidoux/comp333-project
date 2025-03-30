import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
import sys
import json

# ============================
# Step 1: Load Pretrained Models and Vectorizers
# ============================
# Load email text and subject vectorizers and models
text_vectorizer = joblib.load('models/email/email_text_vectorizer.pkl')
subject_vectorizer = joblib.load('models/email/email_subject_vectorizer.pkl')
email_model = joblib.load('models/email/email_model.pkl')

# Load URL feature extraction model
url_model = joblib.load('models/url/url_model.pkl')

# ============================
# Step 2: URL Feature Extraction Function
# ============================
def extract_url_features(url):
    """Extract high-impact features from a given URL."""
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    query_params = parsed_url.query

    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'contains_free': int('free' in url.lower()),
        'contains_win': int(any(word in url.lower() for word in ['win', 'reward', 'gift', 'claim'])),
        'contains_click': int('click' in url.lower()),
        'contains_offer': int('offer' in url.lower()),
        'contains_account': int('account' in url.lower()),
        'contains_auth': int('auth' in url.lower()),
        'contains_login': int('login' in url.lower()),
        'contains_brand': int(any(brand in url.lower() for brand in ['paypal', 'google', 'amazon', 'facebook'])),
        'domain_length': len(domain_parts[-2]) if len(domain_parts) > 1 else 0,
        'subdomain_length': len(domain_parts[0]) if len(domain_parts) > 2 else 0,
        'suspicious_tld': int(domain_parts[-1] in ['top', 'xyz', 'click', 'club', 'biz', 'info', 'work', 'zip', 'mobi']),
        'has_redirect': int('?q=' in url or '?url=' in url or '?redirect=' in url),
        'suspicious_subdomain': int(any(keyword in parsed_url.netloc for keyword in ['auth', 'login', 'secure'])),
        'num_redirects': url.count('http') - 1,
        'path_length': len(parsed_url.path),
        'query_length': len(parsed_url.query),
        'num_query_params': len(query_params),
    }
    return list(features.values())

# ============================
# Step 3: URL Normalization Function
# ============================
def fix_url(url):
    if not re.match(r'^(http://|https://)', url):
        url = 'http://' + url
    if not re.match(r'^(http://www\.|https://www\.)', url):
        url = url.replace('http://', 'http://www.').replace('https://', 'https://www.')
    return url

# ============================
# Step 4: Analyze Email (Text and Subject)
# ============================
def analyze_email(subject, text):
    """Analyze both email subject and text, check for URLs, and classify the email."""
    
    # Extract URLs from both subject and text (if any)
    urls = re.findall(r'https?://\S+|www\.\S+', subject + " " + text)
    urls = [fix_url(url) for url in urls]  # Normalize URLs
    
    # Prepare default probabilities
    url_spam_prob = 0.0

    # ============================
    # Step 1: Analyze Email Subject and Text
    # ============================
    clean_subject = re.sub(r'https?://\S+|www\.\S+', '[URL]', subject)  
    clean_text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)  

    # Vectorize subject and text
    X_subject = subject_vectorizer.transform([clean_subject])
    X_text = text_vectorizer.transform([clean_text])

    # Stack features (important to match training shape)
    X_combined = hstack([X_subject, X_text])

    # subject_prob = email_model.predict_proba(X_subject)[0][1]  # Probability of spam for subject
    # text_prob = email_model.predict_proba(X_text)[0][1]  # Probability of spam for text

    # Predict probabilities
    spam_prob = email_model.predict_proba(X_combined)[0][1]

    
    combined_prob = 0.0  # Default combined probability

    # ============================
    # Step 2: Analyze URL (if any)
    # ============================
    if urls:
        url_features = np.array([extract_url_features(url) for url in urls])
        url_spam_prob = max(url_model.predict_proba(url_features)[:, 1])  

        # ============================
        # Step 3: Combine Results
        # ============================
        combined_prob = 0.3 * spam_prob + 0.7 * url_spam_prob  # Weighted combination

    else:
        combined_prob = spam_prob  # No URLs found, use only email spam probability

    is_spam = combined_prob >= 0.5

    return {
        'spam_prob': round(spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }

# ============================
# Example Usage
# ============================
if __name__ == '__main__':
     # Example text input
    subject = sys.argv[1] if len(sys.argv) > 1 else 'Test'
    text = sys.argv[2] if len(sys.argv) > 2 else 'Good afternoon, when will the test be? Thank you'

    # Predict spam
    prediction_result = analyze_email(subject, text)
    
    # convert mumpy float values to string for JSON serialization
    for key, value in prediction_result.items():
        if isinstance(value, np.float64) or isinstance(value, np.float32):
            prediction_result[key] = round(float(value), 2)

    print(json.dumps(prediction_result, indent=4))
