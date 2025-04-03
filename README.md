# Spam Detection Model

## Spam Detection System

This project is a **Spam Detection System** that leverages advanced machine learning models to classify text inputs—including **SMS, Emails, and YouTube Comments**—as spam or not spam.

The system is built with a **Node.js backend**, which seamlessly integrates **Python-based machine learning models** for accurate predictions. It utilizes our **advanced URL classification model** to extract and analyze URLs separately from the text, assessing their legitimacy before combining the results for a more robust spam detection process.

A frontend interface is also included, allowing users to interact with the system and receive real-time classification results.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [API Endpoints](#api-endpoints)
8. [Technologies Used](#technologies-used)
9. [Report](#report)
10. [Contributors](#contributors)

---

## Features

- **Advanced Spam Detection**: Accurately classifies text inputs—including SMS, emails, YouTube comments, and URLs—as spam or not spam.
- **Specialized Machine Learning Models**: Pre-trained models optimized for different input types.
- **Integrated URL Analysis**: Separates and evaluates URLs within text using a dedicated classification model for enhanced detection.
- **RESTful API**: Efficient backend API for handling real-time prediction requests.
- **User-Friendly Frontend**: Intuitive interface for submitting inputs and viewing results.
- **Comprehensive Data Preprocessing**: Automated tools for cleaning, normalizing, and preparing datasets for training.

---

## Project Structure

📂 spam-detection
├── backend/ - Backend server and dependencies
│ ├── index.js - Node.js backend server
│ ├── package.json - Backend dependencies and scripts
│ ├── public/ - Frontend assets
│ │ ├── css/ - Frontend stylesheets
│ │ ├── js/ - Frontend JavaScript
│ │ └── index.html - Frontend HTML file

├── data/ - Datasets for training and testing
│ ├── comment/ - YouTube comment datasets
│ ├── email/ - Email datasets
│ ├── sms/ - SMS datasets
│ ├── url/ - URL datasets
│ └── normalized/ - Preprocessed datasets

├── models/ - Machine learning models
│ ├── comment/ - YouTube comment spam model
│ ├── email/ - Email spam model
│ ├── sms/ - SMS spam model
│ └── url/ - URL spam model

├── report/ - Final project report
├── tools/ - Dataset preprocessing tools
└── README.md - Project documentation

---

## Datasets

The project uses the following datasets stored in `data/`:

### SMS Spam

- Location: `data/sms/`
- Contains: Labeled SMS messages (spam/ham)
- Format: CSV
- Size: [X] samples

### Email Spam

- Location: `data/email/`
- Contains: Labeled emails (spam/ham)
- Format: CSV
- Size: [X] samples

### YouTube Comments

- Location: `data/comment/`
- Contains: Labeled comments (spam/ham)
- Format: CSV
- Size: [X] samples

### URL Spam

- Location: `data/url/`
- Contains: Labeled URLs (malicious/benign)
- Format: CSV
- Size: [X] samples

Preprocessed versions available in `data/normalized/`

---

## Models

The project employs separate machine learning models optimized for different types of spam classification located in `models/`.

### SMS Spam Model

- Algorithm: Random Forest Classifier
- Features: TF-IDF vectorization (unigrams and bigrams)
- Dataset: SMS spam dataset
- Model File: `models/sms/sms_model.pkl`

### Email Spam Model

- Algorithm: Naive Bayes (MultinomialNB)
- Features: TF-IDF vectorization
- Dataset: Email spam dataset
- Model File: `models/email/email_model.pkl`

### YouTube Comment Spam Model

- Algorithm: Logistic Regression
- Features: Word embeddings with TF-IDF
- Dataset: YouTube comment spam dataset
- Model File: `models/comment/comment_model.pkl`

### URL Spam Model

- Algorithm: XGBoost Classifier
- Features: Extracted URL characteristics such as length, domain type, special characters, and presence of certain keywords.
- Dataset: URL spam dataset
- Model File: `models/url/url_model.pkl`

Each model is pre-trained and saved in the respective directories for easy integration with the backend API.

---

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Node.js (v16 or higher)
- Python (v3.8 or higher)
- pip (latest version)

### Installation Steps

**1.** Clone the Repository

```
git clone https://github.com/yourusername/spam-detection-model.git
cd spam-detection
```

**2.** Backend Setup

```
cd backend
npm install
cd ..
```

**3.** Normalize the data

```
python tools/normalize_data.py
```

**4.** Train the models

```
python/url/train_url_model.py
python/comment/train_comment_model.py
python/sms/train_sms_model.py
python/email/train_email_model.py
```

**5.** Run the backend

```
node backend/index.js
```

**6.** Open `http://localhost:3000/` in your browser.

---

## Usage

Once the backend server is running, you can use the frontend interface or make API requests directly.

### Running the Frontend

Simply open `http://localhost:3000/` in your browser.

### Making API Requests

You can also send requests to the API endpoints via Postman or curl.

Example:

```
curl -X POST http://localhost:3000/predict-sms -H "Content-Type: application/json" -d '{"text": "Win a free iPhone now!"}'
```

Response:

```
{
    "spam_prob": 0.41,
    "url_spam_prob": 0.0,
    "combined_prob": 0.41,
    "is_spam": false
}
```

---

## API Endpoints

### POST /predict-sms

Description: Predicts if a given SMS is spam or not.

Request Body:

```
{
  "text": "Your message here"
}
```

Response:

```
{
    "spam_prob": 0.85,
    "url_spam_prob": 0.0,
    "combined_prob": 0.85,
    "is_spam": true
}
```

### POST /predict-email

Description: Predicts if a given email is spam or not.

Request Body:

```
{
  "subject": "Your subject here",
  "text": "Your message here"
}
```

Response:

```
{
    "spam_prob": 0.75,
    "url_spam_prob": 0.0,
    "combined_prob": 0.75,
    "is_spam": true
}
```

### POST /predict-comment

Description: Predicts if a given YouTube comment is spam or not.

Request Body:

```
{
  "author": "The author here",
  "text": "Your comment here"
}
```

Response:

```
{
    "spam_prob": 0.65,
    "url_spam_prob": 0.0,
    "combined_prob": 0.65,
    "is_spam": true
}
```

---

## Technologies Used

- **Backend** : Node.js, Express.js
- **Frontend** : HTML, CSS, JavaScript
- **Machine Learning** : Python, scikit-learn, XGBoost
- **Data Preprocessing** : Pandas, NumPy

---

## Report

The final project report is available in the `report/` directory. It includes detailed information about the models, datasets, and evaluation metrics used in this project.

---

## Contributors

- Matteo Robidoux
- Raagav Prasanna

For contributions or feedback, please open an issue or submit a pull request!
