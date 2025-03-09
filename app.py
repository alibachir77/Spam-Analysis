from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Load the dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\Spam Analysis\preprocessed_data.csv")

# Handle missing values
df['text'] = df['text'].fillna('')
df['labels'] = df['labels'].fillna('ham')  # Assuming 'ham' is the default category
df['text'] = df['text'].astype(str)

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords and lemmatize
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    
    # Replace numbers with a placeholder tag
    text = re.sub(r'\d+', '<NUM>', text)
    return text

# Apply text preprocessing to the dataset
df['text'] = df['text'].apply(preprocess_text)

# Extract features and labels
X = df['text']
y = df['labels']

# Ensure there are no NaN values
X = X.fillna('')
y = y.fillna('ham')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Support Vector Machine Model
svm = SVC(probability=True)  # Enable probability estimates
svm.fit(X_train_tfidf, y_train)


def predict_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = svm.predict(message_tfidf)[0]
    if svm.probability:
        probability = svm.predict_proba(message_tfidf)[0]
    else:
        probability = None  # Handle case where probability is not available
    return prediction, probability

# Route to preprocess a message
@app.route('/preprocess_message', methods=['POST'])
def preprocess_message():
    data = request.get_json()
    
    if 'message' in data:
        message = data['message']
        
        # Perform text preprocessing on the 'message'
        preprocessed_message = preprocess_text(message)
        
        return jsonify({'preprocessed_message': preprocessed_message})
    else:
        return jsonify({'error': 'Key "message" not found in the input data'})

# Route to predict the category of a preprocessed message
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'preprocessed_message' in data:
        message = data['preprocessed_message']
        
        # Predict using the preprocessed message
        prediction, probability = predict_message(message)
        
        results = {
            'cleaned_message': message,
            'classification': prediction,  # Assuming prediction is 'Spam' or 'Ham'
            'spam_probability': float(probability[1]),  # Probability of being spam
            'ham_probability': float(probability[0])    # Probability of being ham
        }
        
        return jsonify(results)
    else:
        return jsonify({'error': 'Key "preprocessed_message" not found in the input data'})

if __name__ == '__main__':
    app.run(debug=True)
