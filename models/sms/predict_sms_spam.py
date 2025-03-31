import re
import joblib
import numpy as np
from urllib.parse import urlparse
import sys
import json

# Load Pretrained Models and Vectorizer
# Load the trained SMS text vectorizer and model
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load the trained URL spam detection model
url_model = joblib.load('models/url/url_model.pkl')


def extract_url_features(url):
    """Extract relevant features from a URL to detect potential spam."""
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    query_params = parsed_url.query

    # Define URL-based features for spam classification
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
        # Get the highest spam probability if multiple URLs are present
        url_spam_prob = max(url_model.predict_proba(url_features)[:, 1])

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
