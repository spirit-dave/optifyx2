# Email Spam Detector

This project is a web application that classifies emails as either **Spam** or **Not Spam** using a Naive Bayes model trained on text data. The application provides a user-friendly interface for inputting email text and displays real-time classification results and spam statistics.

---

## Project Overview

The application leverages machine learning techniques to classify emails and update real-time statistics. Key components include:

- **Email Classification**: Detects spam and non-spam emails based on text input.
- **Interactive Dashboard**: Displays classification results and real-time spam statistics.
- **Machine Learning Model**: A Naive Bayes classifier trained on email text data using TF-IDF vectorization.

---

## Email Classification Results

### Real-Time Spam Statistics
The application dynamically updates the spam-to-non-spam ratio using a pie chart on the interface. Example outputs:

| Email Text Example       | Prediction |
|--------------------------|------------|
| "Congratulations! You've won!" | Spam       |
| "Meeting scheduled for Monday" | Not Spam  |

### Machine Learning Details:

- **Algorithm**: Naive Bayes Classifier
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset Used**: A cleaned dataset of email messages labeled as spam or non-spam.

---

## Insights from the Analysis

1. **Spam Characteristics**:
   - Emails with excessive promotional language and links are often classified as spam.
   - Keywords like "win," "free," and "offer" heavily influence predictions.

2. **Real-Time Benefits**:
   - Tracking the spam/non-spam ratio helps identify trends in incoming emails.

3. **User Feedback Integration**:
   - Allows for manual correction of classification to refine the model over time.

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (Bootstrap), JavaScript (Chart.js)
- **Machine Learning**: Scikit-learn, Naive Bayes Classifier
- **Data Preprocessing**: TfidfVectorizer

