# Phishing-email-detector
An email phishing detector is a tool or system designed to identify and block phishing emails—fraudulent messages that try to trick users into revealing sensitive information like passwords, credit card numbers, or personal data.

## Objective
Develop a system to detect phishing emails using a combination of cybersecurity principles (threat signatures, blacklists, URL heuristics) and machine learning/NLP for email content analysis.

## Tools Used
1. Python

2. scikit-learn / XGBoost / LightGBM

3. NLTK / spaCy / TextBlob (for text processing)

4. Pandas / NumPy

5. Flask or Streamlit (for web app)

6. Jupyter Notebook (for development)

7. Optional: Email datasets like Kaggle Phishing

## Steps to build
1. Data Collection
Get a dataset of phishing and legitimate emails. Example:

plaintext
Copy
Edit
- Features: Email text, subject, sender, links, etc.
- Labels: phishing (1), legitimate (0)
2. Data Preprocessing
Clean the text (remove HTML tags, URLs, symbols)
Tokenize and normalize text (lowercase, lemmatization)
Remove stopwords
Vectorize text (e.g., TF-IDF or CountVectorizer)
3. Model Building
Split data: train-test (80-20)
Train classifiers:
Logistic Regression
Random Forest
XGBoost
Naive Bayes (works well with text)
Evaluate using: accuracy, precision, recall, F1-score
4. Evaluation
Show confusion matrix
Highlight precision (for low false positives)
Use ROC-AUC curve
5.Deploy as Web App (Optional)
Use Flask or Streamlit to upload and classify email samples
Interface to paste email text and get phishing prediction
## Code
# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# 2. Load Dataset
df = pd.read_csv('data/emails.csv')  # Make sure you have a labeled dataset
df = df[['text', 'label']]  # Columns: 'text' (email body), 'label' (1=phishing, 0=legit)
print(df.head())

# 3. Preprocess Text
def clean_email(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@[\w]*', '', text)  # Remove mentions
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_text'] = df['text'].apply(clean_email)

# 4. Feature Extraction
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Predictions & Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 8. Phishing Detection Function
def detect_phishing(email_text):
    cleaned = clean_email(email_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "⚠️ Phishing" if prediction == 1 else "✅ Legitimate"

# 9. Try it
sample_email = "Dear user, your account is at risk. Please verify your login here: http://fakeurl.com"
print("\nTest Prediction:", detect_phishing(sample_email))

##  Sample Output

![image](https://github.com/user-attachments/assets/28db59e7-94d7-45b6-a77a-9da4b4f75a91)

![image](https://github.com/user-attachments/assets/dd453db6-61b5-4457-9328-eb6b7596a783)


## Features

1.URL feature extraction (suspicious domains, shortening)

2.NLP to detect urgency-based language

3.Integrate with email clients (e.g., Gmail API)

4.Real-time email scanning engine





