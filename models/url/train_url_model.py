from urllib.parse import urlparse, parse_qs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier
from tqdm import tqdm
import time

def extract_url_features(url):
    """Extract high-impact features from a given URL."""
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    query_params = parse_qs(parsed_url.query)

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

print("Loading data...")

# Load URL data
df = pd.read_csv('data/normalized/url_data.csv', keep_default_na=False)

print("Data loaded!\n")

# Extract features from each URL with progress bar
feature_columns = [
    'url_length', 'num_dots',
    'contains_free', 'contains_win', 'contains_click', 'contains_offer',
    'contains_account', 'contains_auth', 'contains_login', 'contains_brand',
    'domain_length', 'subdomain_length', 'suspicious_tld',
    'has_redirect', 'suspicious_subdomain', 'num_redirects',
    'path_length', 'query_length', 'num_query_params'
]
url_features = [extract_url_features(url) for url in tqdm(df['text'], desc="Extracting URL Features")]
url_features_df = pd.DataFrame(url_features, columns=feature_columns)

# Add target variable
url_features_df['is_spam'] = df['is_spam']

# Split data into features and target
X = url_features_df.drop('is_spam', axis=1)
y = url_features_df['is_spam']

# Split data into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define and train the XGBoost model
best_model = XGBClassifier(
    n_estimators=150,           # Fewer trees for faster training
    max_depth=6,                # Balanced complexity
    learning_rate=0.1,          # Reasonable learning rate
    subsample=0.8,              # Sample 80% of data to prevent overfitting
    colsample_bytree=0.8,       # Randomly sample columns to reduce overfitting
    random_state=42,
    eval_metric='logloss',      # Better for binary classification
    early_stopping_rounds=30    # Stop early if no improvement
)

# Train the model
start = time.time()
print("\nStarting training...")
best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")

# Predict and evaluate the model
y_pred = best_model.predict(X_test)
print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, 'models/url/url_model.pkl')

print("\nURL Spam Model saved!")

# Get feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': importances})

# Sort features by importance
print("\nFeature Importance Ranking:")
print(feature_importance_df.sort_values(by='importance', ascending=False))
