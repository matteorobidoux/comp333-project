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

```
📂 spam-detection
├── analysis/ - Analysis scripts and results
│ ├── sms/ - SMS spam analysis results
│ ├── email/ - Email spam analysis results
│ ├── comment/ - YouTube comment spam analysis results
│ ├── url/ - URL spam analysis results
│ └── models/ - Models for analysis
│
├── backend/ - Backend server and dependencies
│ ├── index.js - Node.js backend server
│ ├── package.json - Backend dependencies and scripts
│ └── public/ - Frontend assets
│   ├── css/ - Frontend stylesheets
│   ├── js/ - Frontend JavaScript
│   └── index.html - Frontend HTML file
|
├── data/ - Datasets for training and testing
| ├── analysis/ - Analysis datasets
│ ├── comment/ - YouTube comment datasets
│ ├── email/ - Email datasets
| ├── normalized/ - Preprocessed datasets
│ ├── sms/ - SMS datasets
│ └── url/ - URL datasets
|
├── models/ - Machine learning models
│ ├── comment/ - YouTube comment spam model
│ ├── email/ - Email spam model
│ ├── sms/ - SMS spam model
│ └── url/ - URL spam model
|
├── report/ - Final project report
├── tools/ - Dataset preprocessing tools
└── README.md - Project documentation
```

---

## Datasets

The project uses the following datasets stored in `data/`:

### Complete Dataset Table

| Dataset Type         | Raw Location     | Normalized Location                | Samples | Description                                            | Files                                                                                            |
| -------------------- | ---------------- | ---------------------------------- | ------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **SMS Messages**     | `data/sms/`      | `data/normalized/sms_data.csv`     | 6,307   | Labeled SMS messages                                   | `sms_dataset_1.csv` <br>`sms_dataset_2.csv`                                                      |
| **Emails**           | `data/email/`    | `data/normalized/email_data.csv`   | 17,657  | Labeled emails (spam/ham)                              | `email_dataset_1.csv`<br>`email_dataset_2.csv`<br>`email_dataset_3.csv`<br>`email_dataset_4.csv` |
| **YouTube Comments** | `data/comment/`  | `data/normalized/comment_data.csv` | 1,962   | Labeled YouTube comments (spam/legitimate)             | `comment_dataset.csv`                                                                            |
| **URLs**             | `data/url/`      | `data/normalized/url_data.csv`     | 450,177 | Labeled URLs                                           | `url_dataset.csv`                                                                                |
| **SMS+URL Analysis** | `data/analysis/` | -                                  | 151     | SMS messages with appended URLs for special analysis   | `sms_url_combined.csv`                                                                           |
| **UCI SMS Data**     | -                | `data/normalized/sms_uci_data.csv` | 1,495   | Normalized version of `sms_dataset_2.csv` for analysis | (derived from `sms_dataset_2.csv`)                                                               |

---

## Models

The project employs separate machine learning models optimized for different types of spam classification located in `models/`.

### Spam Detection Models

| Model Type           | Algorithm                    | Features                                 | Dataset                 | Model File Path                    |
| -------------------- | ---------------------------- | ---------------------------------------- | ----------------------- | ---------------------------------- |
| **SMS Spam**         | Random Forest Classifier     | Character-Level TF-IDF                   | SMS spam dataset        | `models/sms/sms_model.pkl`         |
| **Email Spam**       | XGB Classifier               | Subject + Body TF-IDF                    | Email spam dataset      | `models/email/email_model.pkl`     |
| **YouTube Comments** | Gradient Boosting Classifier | Character-Level TF-IDF + Author Analysis | YouTube comment dataset | `models/comment/comment_model.pkl` |
| **URL Spam**         | Light GBM                    | URL Feature Engineering                  | URL spam dataset        | `models/url/url_model.pkl`         |

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
cd spam-detection-model
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
- **Machine Learning** : Python, scikit-learn,XGBoost, LightGBM
- **Data Preprocessing** : Pandas, NumPy

---

## Report

The final project report is available in the `report/` directory. It includes detailed information about the project, including the methodology, results, and conclusions drawn from the analysis.

---

## Contributors

- Matteo Robidoux
- Raagav Prasanna

For contributions or feedback, please open an issue or submit a pull request!
