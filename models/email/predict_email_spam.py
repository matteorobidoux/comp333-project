import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack
import sys
import json

# Load Pretrained Models and Vectorizers
# Load vectorizers and models for email spam classification
text_vectorizer = joblib.load('models/email/email_text_vectorizer.pkl')
subject_vectorizer = joblib.load('models/email/email_subject_vectorizer.pkl')
email_model = joblib.load('models/email/email_model.pkl')

# Load the URL feature extraction model
url_model = joblib.load('models/url/url_model.pkl')


def extract_url_features(url):
    """Extract features from a URL to determine spam probability."""
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    query_params = parsed_url.query

    # Define URL features for classification
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


def fix_url(url):
    """Ensure URLs have the correct protocol and format."""
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
        # Predict spam probability for the URLs
        url_spam_prob = max(url_model.predict_proba(url_features)[:, 1])
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
