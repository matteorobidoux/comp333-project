import time
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.metrics import classification_report

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

start = time.time()

# Train LightGBM model with proper early stopping
model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    valid_names=['valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")

y_pred = (model.predict(X_test) > 0.5).astype(int)
y_pred_proba = model.predict(X_test)  # Probability scores

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam'])

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Generate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# --- VISUALIZATION ---
plt.figure(figsize=(18, 6))
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# 1. ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# 2. Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {avg_precision:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

# 3. Confusion Matrix
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Spam'], 
            yticklabels=['Legitimate', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.savefig('analysis/url/url_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PRINT METRICS ---
print(f"\n{' MODEL EVALUATION ':=^60}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))

# --- SAVE MODEL ---
joblib.dump(model, 'models/url/url_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/url/url_vectorizer.pkl')

print("\nModel and tokenizer saved!")