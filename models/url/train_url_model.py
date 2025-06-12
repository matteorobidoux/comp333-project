import time
import pandas as pd
import numpy as np
import re
import tldextract
import joblib
import lightgbm as lgb
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve, classification_report)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap

# --- Load Data ---
df = pd.read_csv('data/normalized/url_data.csv', keep_default_na=False)
y = df['is_spam'].values
raw_urls = df['text'].astype(str)

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

print("Extracting features...")
feature_data = [extract_features(url) for url in tqdm(urls, desc="Feature Extraction")]
feature_df = pd.DataFrame(feature_data)

# --- TF-IDF Vectorization ---
print("\nGenerating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(urls)

engineered_feature_names = [
    'url_length', 'num_digits', 'num_special_chars', 'entropy_url',
    'num_subdomains', 'domain_length', 'path_length', 'query_length',
    'num_query_params', 'tld_length', 'suspicious_tld', 'has_ip_address',
    'contains_free', 'contains_win', 'contains_login', 'contains_auth',
    'contains_account', 'contains_offer', 'contains_secure', 'num_dots',
    'num_hyphens', 'num_slashes'
]

# Clean feature names for TF-IDF
def clean_name(s):
    return re.sub(r'[\"\'\\\n\r\t]', '', s)

# Set TF-IDF feature names
tfidf_feature_names = [f"TFIDF_{clean_name(f)}" for f in tfidf_vectorizer.get_feature_names_out()]

# Combine all feature names
all_feature_names = engineered_feature_names + tfidf_feature_names

# Combine all features
X = hstack([feature_df.values, tfidf_matrix]).tocsr()

# --- Cross-Validation Setup ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

accuracies, aucs, avg_precisions = [], [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

start = time.time()
print("\nStarting Stratified K-Fold training...\n")

fold = 1
for train_index, test_index in kf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=all_feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=all_feature_names)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)]
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Avg Precision: {avg_precision:.4f}")

    accuracies.append(accuracy)
    aucs.append(auc)
    avg_precisions.append(avg_precision)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_pred_proba)

    fold += 1

print("Training completed!")
end = time.time()
total_time = end - start
print(f"\nTotal training time: {total_time:.2f} seconds")


# --- Final Evaluation ---
final_cm = confusion_matrix(all_y_true, all_y_pred)
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_proba)
final_auc = roc_auc_score(all_y_true, all_y_proba)
final_avg_prec = average_precision_score(all_y_true, all_y_proba)
final_accuracy = accuracy_score(all_y_true, all_y_pred)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# --- ROC Curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {final_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('analysis/url/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {final_avg_prec:.3f}')
plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.savefig('analysis/url/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('analysis/url/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Clean Feature Importance Plot (Top 10) ---
print("\nGenerating clean feature importance plot...")

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': model.feature_importance(importance_type='gain')
})

# Remove the highest one
top_features = importance_df.sort_values('Importance', ascending=False).head(21)
top_features = top_features.iloc[1:]  # Remove top feature

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features,
    palette='viridis'
)
plt.title('Top 20 Important Features', fontsize=14)
plt.xlabel('Importance Score (Gain)', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.savefig('analysis/url/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Summary ---
print(f"\n{' FINAL EVALUATION (Stratified K-Fold) ':=^60}\n")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average AUC-ROC: {np.mean(aucs):.4f}")
print(f"Average Precision: {np.mean(avg_precisions):.4f}\n")
print("Classification Report (aggregated predictions):")
print(classification_report(all_y_true, all_y_pred, target_names=['Benign', 'Malicious']))

# --- Save Model and Vectorizer ---
joblib.dump(model, 'models/url/url_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/url/url_vectorizer.pkl')

print("\nModel and vectorizer saved!")

# --- Overall Performance Summary ---
print(f"\n{' FINAL CROSS-VALIDATION RESULTS ':=^60}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Mean AUC: {np.mean(aucs):.4f}")
print(f"Mean Average Precision: {np.mean(avg_precisions):.4f}")