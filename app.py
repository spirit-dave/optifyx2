from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Specify the new template folder
app = Flask(__name__, template_folder="C:/Users/DUST/custom_templates")

# Step 1: Train the model if not already trained
def train_model():
    # Load dataset
    data = pd.read_csv('cleaned_spam.csv')

    # Ensure the 'message' column exists
    if 'message' not in data.columns:
        raise KeyError("The dataset does not contain a 'message' column for text data.")

    # Rename 'message' to 'text' for consistency
    data['text'] = data['message']

    # Ensure the 'label' column exists
    if 'label' not in data.columns:
        raise KeyError("The dataset does not contain a 'label' column for spam labels.")

    # Preprocess text
    data['text'] = data['text'].str.lower()  # Convert to lowercase
    data['text'] = data['text'].str.replace(r'\W', ' ', regex=True)  # Remove special characters

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate and save the model
    y_pred = model.predict(X_test)
    print(f"Training Accuracy: {accuracy_score(y_test, y_pred)}")
    joblib.dump(model, 'spam_detector_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model and vectorizer saved.")

# Train the model (uncomment if needed)
try:
    joblib.load('spam_detector_model.pkl')
    print("Model already exists.")
except FileNotFoundError:
    print("Training model...")
    train_model()

# Step 2: Load model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stats = {"spam": 0, "not_spam": 0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        email_text = request.form['email_text']

        # Preprocess and predict
        vectorized_text = vectorizer.transform([email_text])
        prediction = model.predict(vectorized_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"

        # Update stats
        stats["spam" if prediction == 1 else "not_spam"] += 1

        return render_template('index.html', email_text=email_text, result=result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
