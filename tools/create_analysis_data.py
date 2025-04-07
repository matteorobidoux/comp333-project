import pandas as pd

sms_data = pd.read_csv("data/normalized/sms_uci_data.csv")
url_data = pd.read_csv("data/normalized/url_data.csv")

# Step 1: Keep only 20% of the SMS data with equal representation of is_spam
spam_data = sms_data[sms_data['is_spam'] == 1]
non_spam_data = sms_data[sms_data['is_spam'] == 0]

spam_sample = spam_data.sample(frac=0.5, random_state=42)
non_spam_sample = non_spam_data.sample(frac=0.5, random_state=42)

balanced_sms_data = pd.concat([spam_sample, non_spam_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Randomly sample an equal number of rows from the URL data
num_samples = len(balanced_sms_data)
spam_url_data = url_data[url_data['is_spam'] == 1]
non_spam_url_data = url_data[url_data['is_spam'] == 0]

spam_url_sample = spam_url_data.sample(n=num_samples // 2, random_state=42)
non_spam_url_sample = non_spam_url_data.sample(n=num_samples // 2, random_state=42)

balanced_url_data = pd.concat([spam_url_sample, non_spam_url_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Concatenate the URL text to the SMS text
balanced_sms_data['text'] = balanced_sms_data['text'] + ' ' + balanced_url_data['text']

# Step 4: Update the is_spam column based on the logic
balanced_sms_data['is_spam'] = balanced_sms_data['is_spam'] | balanced_url_data['is_spam']

# Save the updated dataset to a new CSV file
balanced_sms_data.to_csv("data/analysis/sms_url_combined.csv", index=False)

print("Combined SMS and URL dataset created and saved to 'data/analysis/sms_url_combined.csv'.")