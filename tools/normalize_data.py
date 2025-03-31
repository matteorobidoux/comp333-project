import pandas as pd
import re

def normalize_text(text):
    """Normalize a single text value by removing unwanted characters and trimming spaces."""
    if pd.isna(text):  # Handle NaN values
        text = ''
    text = re.sub(r'�', '', str(text)).strip()
    return text

def normalize_text_column(data, column_names):
    """Normalize text in specified columns using the normalize_text function."""
    for column in column_names:
        data[column] = data[column].apply(normalize_text)
    return data

def extract_subject(text):
    """Extract and remove the subject line from email text."""
    text = normalize_text(text)  # Ensure the text is normalized before processing
    return re.sub(r'Subject: .*?(\n|$)', '', text, flags=re.IGNORECASE).strip()

def remove_urls(text):
    """Remove URLs from text."""
    text = normalize_text(text)  # Ensure the text is normalized before processing
    return re.sub(r'https?://\S+|www\.\S+', '', text).strip()

def change_column_names_and_data(data, column_mapping):
    """Rename columns and normalize text and target values."""
    data = data.rename(columns=column_mapping)
    data['is_spam'] = data['is_spam'].map({'ham': 0, 'spam': 1, 0: 0, 1: 1})
    return data

def get_email_data(email_data):
    """Process and return cleaned email data."""
    email_data = email_data[['label', 'text']]
    email_data = change_column_names_and_data(email_data, {'label': 'is_spam', 'text': 'text'})
    email_data['subject'] = email_data['text'].apply(extract_subject)
    email_data['text'] = email_data['text'].apply(remove_urls)
    email_data = normalize_text_column(email_data, ['text', 'subject'])
    return email_data

def get_sms_data(sms_data):
    """Process and return cleaned SMS data."""
    sms_data = sms_data[['v1', 'v2']]
    sms_data = change_column_names_and_data(sms_data, {'v1': 'is_spam', 'v2': 'text'})
    sms_data['text'] = sms_data['text'].apply(remove_urls)
    sms_data = normalize_text_column(sms_data, ['text'])
    return sms_data

def get_url_data(url_data):
    """Process and return cleaned URL data."""
    url_data = url_data[['result', 'url']]
    url_data = change_column_names_and_data(url_data, {'result': 'is_spam', 'url': 'text'})
    url_data = normalize_text_column(url_data, ['text'])
    return url_data

def get_comments_data(comments_data):
    """Process and return cleaned comment data."""
    comments_data = comments_data[['CLASS', 'CONTENT', 'AUTHOR']]
    comments_data = change_column_names_and_data(comments_data, {'CLASS': 'is_spam', 'CONTENT': 'text', 'AUTHOR': 'author'})
    comments_data['text'] = comments_data['text'].apply(remove_urls)
    comments_data = normalize_text_column(comments_data, ['text', 'author'])
    return comments_data

def combine_dataframes(dataframes):
    """Combine multiple DataFrames into one."""
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

def save_data(data, output_file):
    """Save DataFrame to a CSV file."""
    data.to_csv(output_file, index=False)

# Load and process datasets
email_data = [get_email_data(pd.read_csv(f'data/email/email_dataset_{i}.csv')) for i in range(1, 5)]
url_data = get_url_data(pd.read_csv('data/url/url_dataset.csv'))
comment_data = get_comments_data(pd.read_csv('data/comment/comment_dataset.csv'))
sms_data = [get_sms_data(pd.read_csv(f'data/sms/sms_dataset_{i}.csv')) for i in range(1, 3)]

email_data = combine_dataframes(email_data)
sms_data = combine_dataframes(sms_data)

# Save processed data
save_data(email_data, 'data/normalized/email_data.csv')
save_data(comment_data, 'data/normalized/comment_data.csv')
save_data(sms_data, 'data/normalized/sms_data.csv')
save_data(url_data, 'data/normalized/url_data.csv')

print("Data processing and saving complete!")