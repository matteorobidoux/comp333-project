import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from scipy.sparse import hstack
from urllib.parse import urlparse, parse_qs
import tldextract
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load Pretrained SMS Model and Vectorizer
text_vectorizer = joblib.load('models/sms/sms_text_vectorizer.pkl')
sms_model = joblib.load('models/sms/sms_model.pkl')

# Load Pretrained URL Model and Vectorizer
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

def predict_url_spam(url):
    url = strip_scheme(url)
    structured_features = np.array(extract_url_features(url)).reshape(1, -1)
    tfidf_features = url_vectorizer.transform([url])
    combined_features = hstack([structured_features, tfidf_features])
    prediction = url_model.predict(combined_features)
    return int(prediction[0] > 0.5)

def extract_url_from_text(text):
    url_pattern = r'https?://\S+|www\.\S+'
    match = re.search(url_pattern, text)
    if match:
        url = match.group(0)
        text_without_url = re.sub(url_pattern, '', text).strip()
        return url, text_without_url
    return None, text

def evaluate_hybrid_model(file_path):
    data = pd.read_csv(file_path, keep_default_na=False)
    y_true = data['is_spam'].values
    texts = data['text'].astype(str).values

    sms_predictions = []
    url_predictions = []

    for text in texts:
        url, clean_text = extract_url_from_text(text)

        X_sms = text_vectorizer.transform([clean_text])
        sms_pred = sms_model.predict(X_sms)[0]
        sms_predictions.append(sms_pred)

        if url:
            try:
                url_pred = predict_url_spam(url)
            except Exception as e:
                print(f"Error processing URL: {url}, Error: {e}")
                url_pred = 0
        else:
            url_pred = 0
        url_predictions.append(url_pred)

    combined_predictions = [
        1 if sms == 1 or url == 1 else 0
        for sms, url in zip(sms_predictions, url_predictions)
    ]

    accuracy = accuracy_score(y_true, combined_predictions)
    report = classification_report(y_true, combined_predictions)

    print(f"\nHybrid Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Visualization
    cm = confusion_matrix(y_true, combined_predictions)
    auc = roc_auc_score(y_true, combined_predictions)
    avg_prec = average_precision_score(y_true, combined_predictions)
    fpr, tpr, _ = roc_curve(y_true, combined_predictions)
    precision, recall, _ = precision_recall_curve(y_true, combined_predictions)

    # Visualization
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12

    # --- ROC Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('analysis/sms/hybrid/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#e74c3c', lw=2, label=f'AP = {avg_prec:.3f}')
    plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('analysis/sms/hybrid/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('analysis/sms/hybrid/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    file_path = 'data/analysis/sms_url_combined.csv'
    evaluate_hybrid_model(file_path)
