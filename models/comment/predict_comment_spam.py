import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack
import sys
import json
import tldextract
from urllib.parse import parse_qs
from scipy.special import expit

# Load pre-trained models and vectorizers
text_vectorizer = joblib.load('models/comment/comment_text_vectorizer.pkl')
author_vectorizer = joblib.load('models/comment/comment_author_vectorizer.pkl')
comments_model = joblib.load('models/comment/comment_model.pkl')
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

def fix_url(url):
    """Fix the URL format to ensure it starts with http://www or https://www"""
    url = strip_scheme(url)
    if not re.match(r'^(http://|https://)', url):
        url = 'http://' + url
    if not re.match(r'^(http://www\.|https://www\.)', url):
        url = url.replace('http://', 'http://www.').replace('https://', 'https://www.')
    return url

def analyze_comment(author, text):
    """Analyze both comments author and text, check for URLs, and classify the comments."""
    
    # Extract URLs from both author and text (if any)
    urls = re.findall(r'https?://\S+|www\.\S+', author + " " + text)
    urls = [fix_url(url) for url in urls]  # Normalize URLs
    
    # Prepare default probabilities
    url_spam_prob = 0.0

    # Clean author and text by removing URLs
    clean_author = re.sub(r'https?://\S+|www\.\S+', '', author)  
    clean_text = re.sub(r'https?://\S+|www\.\S+', '', text)  

    # Vectorize author and text
    X_author = author_vectorizer.transform([clean_author])
    X_text = text_vectorizer.transform([clean_text])

    # Stack features (important to match training shape)
    X_combined = hstack([X_author, X_text])

    # Predict probabilities
    spam_prob = comments_model.predict_proba(X_combined)[0][1]

    combined_prob = 0.0  

    # If URLs are found, extract features and predict spam probability
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
        
        # Combine probabilities with weights
        combined_prob = 0.4 * spam_prob + 0.6 * url_spam_prob  # Weighted combination

    else:
        # If no URLs are found, use only the comments spam probability
        combined_prob = spam_prob  # No URLs found, use only comments spam probability

    # Determine if the comment is spam based on combined probability
    is_spam = combined_prob >= 0.5

    return {
        'spam_prob': round(spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }

if __name__ == '__main__':
    # Predefine author and text if not provided as command line arguments
    author = sys.argv[1] if len(sys.argv) > 1 else 'example_author'
    text = sys.argv[2] if len(sys.argv) > 2 else 'Sick vid bro'

    # Predict spam
    prediction_result = analyze_comment(author, text)

    # Convert numpy float values to string for JSON serialization
    for key, value in prediction_result.items():
        if isinstance(value, (np.float64, np.float32)):
            prediction_result[key] = round(float(value), 2)

    # Print the result as a JSON string
    print(json.dumps(prediction_result, indent=4))