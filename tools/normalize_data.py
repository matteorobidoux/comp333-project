import pandas as pd
import re

# ============================
# Helper Functions
# ============================

def normalize_text_column(data, column_name):
    """Normalize a text column by cleaning and standardizing text."""
    # Remove special characters and strip whitespaces
    data[column_name] = data[column_name].replace(r'�', '', regex=True).str.strip()

    # Fill missing values with an empty string
    data[column_name] = data[column_name].fillna('').astype(str)
    return data

def extract_subject(text):
    """Extract and remove the subject line from email text."""
    return re.sub(r'Subject: .*?(\n|$)', '', text, flags=re.IGNORECASE).strip()

def preprocess_text(text):
    """Clean and preprocess text for SMS data."""
    return re.sub(r'https?://\S+|www\.\S+', '[URL]', text).strip().lower()

def change_column_names_and_data(data, column_mapping):
    """Rename columns and map values in a DataFrame."""
    data = data.rename(columns=column_mapping)

    # Normalize text columns and map target column (is_spam)
    data = normalize_text_column(data, 'text')
    data['is_spam'] = data['is_spam'].map({'ham': 0, 'spam': 1, 0: 0, 1: 1})
    
    return data

# ============================
# Data Loading and Processing Functions
# ============================

def get_email_data(email_data):
    """Process and return cleaned email data."""
    email_data = email_data[['label', 'text']]
    email_data = change_column_names_and_data(email_data, {'label': 'is_spam', 'text': 'text'})

    # Extract the subject for email data
    email_data['subject'] = email_data['text'].apply(extract_subject)
    
    return email_data

def get_sms_data(sms_data):
    """Process and return cleaned SMS data."""
    sms_data = sms_data[['v1', 'v2']]
    sms_data = change_column_names_and_data(sms_data, {'v1': 'is_spam', 'v2': 'text'})

    # Replace URLs with [URL] in SMS texts
    sms_data['text'] = sms_data['text'].apply(preprocess_text)
    
    return sms_data

def get_url_data(url_data):
    """Process and return cleaned URL data."""
    url_data = url_data[['result', 'url']]
    url_data = change_column_names_and_data(url_data, {'result': 'is_spam', 'url': 'text'})
    
    return url_data

def get_comments_data(comments_data):
    """Process and return cleaned comment data."""
    comments_data = comments_data[['CLASS', 'CONTENT', 'AUTHOR']]
    comments_data = change_column_names_and_data(comments_data, {'CLASS': 'is_spam', 'CONTENT': 'text', 'AUTHOR': 'author'})
    

    comments_data['text'] = comments_data['text'].apply(preprocess_text)

    return comments_data

# ============================
# Data Combination and Saving
# ============================

def combine_datasets(datasets):
    """Combine multiple datasets into a single DataFrame."""
    return pd.concat(datasets, ignore_index=True)

def save_data(data, output_file):
    """Save DataFrame to a CSV file."""
    data.to_csv(output_file, index=False)

# ============================
# Main Processing Workflow
# ============================

# Load datasets
# email_datasets = [get_email_data(pd.read_csv(f'../data/email/email_dataset_{i}.csv')) for i in range(1, 5)]
url_data = get_url_data(pd.read_csv('data/url/url_dataset.csv'))
comments_data = get_comments_data(pd.read_csv('data/comments/comments_dataset.csv'))

# Combine email datasets
# combined_email_data = combine_datasets(email_datasets)
combined_comments_data = combine_datasets([comments_data])

# Combine URL data (if you have more URL datasets, you can combine them similarly)
combined_url_data = combine_datasets([url_data])

# Save cleaned and combined data
# save_data(combined_email_data, '../data/normalized/email_data.csv')
save_data(combined_comments_data, 'data/normalized/comments_data.csv')
save_data(combined_url_data, 'data/normalized/url_data.csv')

print("✅ Data processing and saving complete!")
