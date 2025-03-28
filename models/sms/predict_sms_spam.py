import re
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# ============================
# Step 1: Load Pretrained Models and Vectorizer
# ============================
# Load SMS text vectorizer and model
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

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
        # Basic URL features
        'url_length': len(url),
        'num_dots': url.count('.'),

        # Keyword-based features
        'contains_free': int('free' in url.lower()),
        'contains_win': int(any(word in url.lower() for word in ['win', 'reward', 'gift', 'claim'])),
        'contains_click': int('click' in url.lower()),
        'contains_offer': int('offer' in url.lower()),
        'contains_account': int('account' in url.lower()),
        'contains_auth': int('auth' in url.lower()),
        'contains_login': int('login' in url.lower()),
        'contains_brand': int(any(brand in url.lower() for brand in ['paypal', 'google', 'amazon', 'facebook'])),

        # Domain and subdomain features
        'domain_length': len(domain_parts[-2]) if len(domain_parts) > 1 else 0,
        'subdomain_length': len(domain_parts[0]) if len(domain_parts) > 2 else 0,
        'suspicious_tld': int(domain_parts[-1] in ['top', 'xyz', 'click', 'club', 'biz', 'info', 'work', 'zip', 'mobi']),

        # Redirect and suspicious path features
        'has_redirect': int('?q=' in url or '?url=' in url or '?redirect=' in url),
        'suspicious_subdomain': int(any(keyword in parsed_url.netloc for keyword in ['auth', 'login', 'secure'])),
        'num_redirects': url.count('http') - 1,

        # Path and query features
        'path_length': len(parsed_url.path),
        'query_length': len(parsed_url.query),
        'num_query_params': len(query_params),
    }
    return list(features.values())

# ============================
# Step 2: URL Normalization Function
# ============================
def fix_url(url):
        # Check if the URL has a valid scheme (http or https), and ensure it starts with 'http://www.' or 'https://www.'
        if not re.match(r'^(http://|https://)', url):
            url = 'http://' + url  # Default to http if no scheme is provided
        if not re.match(r'^(http://www\.|https://www\.)', url):
            # If the URL doesn't start with 'www.', add it
            url = url.replace('http://', 'http://www.').replace('https://', 'https://www.')
        return url


# ============================
# Step 3: URL and Text Analysis
# ============================
def analyze_text(text):
    """Check for URLs, analyze them, and classify the SMS text."""
    # Extract URL from the text (if any)
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    urls = [fix_url(url) for url in urls]  # Normalize URLs
    
    # Prepare default predictions
    sms_spam_prob = 0.0
    url_spam_prob = 0.0

    # ============================
    # Step 4: Analyze SMS Text
    # ============================
    # Clean and vectorize text for SMS analysis
    clean_text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)  # Replace URLs with placeholder
    X_text = text_vectorizer.transform([clean_text])

    # Get SMS model prediction probability
    sms_spam_prob = sms_model.predict_proba(X_text)[0][1]

    # ============================
    # Step 5: Analyze URL (if any)
    # ============================
    if urls:
        # Extract features from all URLs found
        url_features = np.array([extract_url_features(url) for url in urls])

        # Predict URL spam probability
        url_spam_prob = max(url_model.predict_proba(url_features)[:, 1])  # Max risk if multiple URLs

        # ============================
        # Step 6: Combine SMS and URL Results
        # ============================
        # Weighted combination of probabilities
        combined_prob = 0.7 * sms_spam_prob + 0.3 * url_spam_prob

    else:
        # No URLs found, use only SMS spam probability
        combined_prob = sms_spam_prob

    # Decision threshold for spam classification
    is_spam = combined_prob >= 0.5

    return {
        'sms_spam_prob': round(sms_spam_prob, 4),
        'url_spam_prob': round(url_spam_prob, 4),
        'combined_prob': round(combined_prob, 4),
        'is_spam': bool(is_spam)
    }

# ============================
# Step 7: Main Prediction Function
# ============================
def predict_spam(text):
    """Wrapper function to analyze input text and predict spam."""
    result = analyze_text(text)
    
    print(f"✅ SMS Spam Probability: {result['sms_spam_prob']}")
    print(f"🔗 URL Spam Probability: {result['url_spam_prob']}")
    print(f"⚡ Combined Spam Probability: {result['combined_prob']}")
    print(f"🚨 Final Decision: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")

    return result

# ============================
# Example Usage
# ============================
if __name__ == '__main__':
    sample_texts = [
        "Congratulations! You've won a free iPhone. Click here: https://win-free-rewards.biz to claim your prize!",
        "Hi, please check our latest product on https://www.amazon.com/secure-shop now.",
        "Secure your account by logging in at https://auth.secure-login.net",
        "Hello, just checking in to see how you're doing! No links here :)",
        # Nigerian prince scam
        "Dear friend, I am a Nigerian prince and I need your help to transfer $10 million. Please send me your bank details.",
        "Get paid to work from home! Visit https://www.work-from-home.com for more info.",
        # Phishing
        "Your PayPal account has been compromised. Please verify your identity at https://paypal.com/verify-account.",
        "GENT! We are trying to contact you. Last weekends draw shows that you won a �1000 prize GUARANTEED. Call 09064012160. Claim Code K52. Valid 12hrs only. 150ppm"
    ]

    for text in sample_texts:
        print("\n===============================")
        print(f"📩 Analyzing Text: {text}")
        predict_spam(text)