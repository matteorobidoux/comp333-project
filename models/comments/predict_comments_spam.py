import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# ============================
# Step 1: Load Pretrained Models and Vectorizers
# ============================
# Load comments text and author vectorizers and models
text_vectorizer = joblib.load('models/comments/comments_text_vectorizer.pkl')
author_vectorizer = joblib.load('models/comments/comments_author_vectorizer.pkl')
comments_model = joblib.load('models/comments/comments_model.pkl')

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
# Step 4: Analyze comments (Text and author)
# ============================
def analyze_comments(author, text):
    """Analyze both comments author and text, check for URLs, and classify the comments."""
    
    # Extract URLs from both author and text (if any)
    urls = re.findall(r'https?://\S+|www\.\S+', author + " " + text)
    urls = [fix_url(url) for url in urls]  # Normalize URLs
    
    # Prepare default probabilities
    url_spam_prob = 0.0

    # ============================
    # Step 1: Analyze comments author and Text
    # ============================
    clean_author = re.sub(r'https?://\S+|www\.\S+', '[URL]', author)  
    clean_text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)  

    # Vectorize author and text
    X_author = author_vectorizer.transform([clean_author])
    X_text = text_vectorizer.transform([clean_text])

    # Stack features (important to match training shape)
    X_combined = hstack([X_author, X_text])

    # author_prob = comments_model.predict_proba(X_author)[0][1]  # Probability of spam for author
    # text_prob = comments_model.predict_proba(X_text)[0][1]  # Probability of spam for text

    # Predict probabilities
    spam_prob = comments_model.predict_proba(X_combined)[0][1]

    
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
        combined_prob = 0.4 * spam_prob + 0.6 * url_spam_prob  # Weighted combination

    else:
        combined_prob = spam_prob  # No URLs found, use only comments spam probability

    is_spam = combined_prob >= 0.5

    return {
        'comments_spam_prob': round(spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }

# ============================
# Step 9: Main Prediction Function
# ============================
def predict_spam(author, text):
    """Wrapper function to analyze comments author and text and predict spam."""
    result = analyze_comments(author, text)
    
    print(f"✅ Comment Spam Probability: {result['comments_spam_prob']}")
    print(f"🔗 URL Spam Probability: {result['url_spam_prob']}")
    print(f"⚡ Combined Spam Probability: {result['combined_prob']}")
    print(f"🚨 Final Decision: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")

    return result

# ============================
# Example Usage
# ============================
if __name__ == '__main__':
    sample_commentss =  [
    {
        'author': 'JohnDoe123',
        'text': 'Check out my amazing YouTube channel! Don’t forget to subscribe!'
    },
    {
        'author': 'JaneSmith',
        'text': 'This is a legitimate comment without any spammy links or promotions.'
    },
    {
        'author': 'SpamBot99',
        'text': 'Win a free iPhone! Click here: http://malicious-site.com/free-iphone-offer'
    },
    {
        'author': 'TrustedUser789',
        'text': 'I love this product! Definitely going to buy it! No links or spam here.'
    },
    {
        'author': 'FreeMoney2025',
        'text': 'Congratulations! You’ve won $1000! Claim your prize at http://dangerous-link.com/claim-your-prize'
    },
    {
        'author': 'LegitCommenter',
        'text': 'Great video! Keep up the good work!'
    },
    {
        'author': 'ClickHereForFreeStuff',
        'text': 'Hurry up and get your free gift now: https://phishing-site.com/free-gift-now'
    },
    {
        'author': 'User12345',
        'text': 'Amazing tutorial, learned a lot from it! Keep posting helpful content.'
    },
    {
        'author': 'ScamAlertBot',
        'text': 'You have been selected for a free giveaway! Click here to claim: http://malicious-link.com/free-giveaway'
    },
    {
        'author': 'NoSpamHere',
        'text': 'Great video, I’ve been following your channel for a while! Keep it up.'
    }
]

    for comments in sample_commentss:
        print("\n===============================")
        print(f"📧 Analyzing comments: {comments['author']}")
        predict_spam(comments['author'], comments['text'])
