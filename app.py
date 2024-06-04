from flask import Flask, request, render_template
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('logistic_model.pkl')
count_vectorizer = joblib.load('count_vectorizer.pkl')
tfidf_transformer = joblib.load('ttfidf_transformer.pkl')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = clean_text(message)
        vect = count_vectorizer.transform([cleaned_message])
        tfidf = tfidf_transformer.transform(vect)
        prediction = model.predict(tfidf)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('result.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)