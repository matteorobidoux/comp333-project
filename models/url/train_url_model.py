# Import necessary libraries
from urllib.parse import urlparse, parse_qs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier
from tqdm import tqdm
import time



# ============================
# Function to Extract URL Features
# ============================
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

# ============================
# Step 1: Load and Preprocess Data
# ============================
# Load URL data
df = pd.read_csv('data/normalized/url_data.csv')

# Extract features from each URL
feature_columns = [
    'url_length', 'num_dots', 
    'contains_free', 'contains_win', 'contains_click', 'contains_offer',
    'contains_account', 'contains_auth', 'contains_login', 'contains_brand',
    'domain_length', 'subdomain_length', 'suspicious_tld',
    'has_redirect', 'suspicious_subdomain', 'num_redirects',
    'path_length', 'query_length', 'num_query_params'
]

# Apply feature extraction with progress bar
url_features = [extract_url_features(url) for url in tqdm(df['text'], desc="Extracting URL Features")]
url_features_df = pd.DataFrame(url_features, columns=feature_columns)

# Add target variable
url_features_df['is_spam'] = df['is_spam']

# ============================
# Step 2: Split Data for Training and Testing
# ============================
X = url_features_df.drop('is_spam', axis=1)
y = url_features_df['is_spam']

# Split data into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Step 3: Hyperparameter Tuning
# ============================
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
}

# Initialize and run RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit model to training data

start = time.time()
print("Starting training...")
print("start time:", start)
randomized_search.fit(X_train, y_train)
end = time.time()
print("Training completed.")
print("end time:", end)
print("Training time:", end - start)


# Retrieve the best model
best_model = randomized_search.best_estimator_

# ============================
# Step 4: Model Evaluation
# ============================
# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
print(f'✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Feature Importance
# ============================
# Save the trained model
joblib.dump(best_model, 'models/url/url_model.pkl')

# Print the best hyperparameters
print("🏆 Best Hyperparameters:", randomized_search.best_params_)

# Get feature importance from the trained model
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': importances})

# Print sorted feature importance
print("🔍 Feature Importance Ranking:")
print(feature_importance_df.sort_values(by='importance', ascending=False))
