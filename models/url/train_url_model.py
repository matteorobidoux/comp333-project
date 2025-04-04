import pandas as pd
import numpy as np
import re
import tldextract
import joblib
import lightgbm as lgb
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from tqdm import tqdm

# Load dataset
df = pd.read_csv('data/normalized/url_data.csv', keep_default_na=False)

# Extract target variable
y = df['is_spam'].values
urls = df['text'].astype(str).values  # Ensure URLs are strings

# --- FEATURE ENGINEERING ---
def extract_features(url):
    """Extract extensive set of URL features."""
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

# Extract structured features
feature_data = [extract_features(url) for url in tqdm(urls, desc="Extracting Features")]
feature_df = pd.DataFrame(feature_data)

# --- TF-IDF ON URL TEXT ---
print("\nGenerating TF-IDF features...")

tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=2000)
tfidf_matrix = tfidf_vectorizer.fit_transform(urls)

# Combine structured and tfidf features
X_combined = hstack([feature_df.values, tfidf_matrix])

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)

# --- LIGHTGBM TRAINING ---
print("\nTraining LightGBM Model...")

# Add an evaluation metric (binary_logloss or auc)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',  # ✅ Required for early stopping
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train LightGBM model with proper early stopping
model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    valid_names=['valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# --- MODEL EVALUATION ---
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- SAVE MODEL ---
joblib.dump(model, 'models/url/url_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/url/url_vectorizer.pkl')

print("\nModel and tokenizer saved!")