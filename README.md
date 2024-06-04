# Twitter-Sentiment-Analysis
This project implements a Twitter Sentiment Analysis web application using Flask. The application allows users to input a tweet and predicts whether the sentiment of the tweet is positive or negative.
### Explanation

- **Overview**:
   This project uses the Sentiment140 dataset to train a machine learning model for sentiment analysis. The model is built using Logistic Regression and is deployed as a web application using Flask..
- **Features**:
  - Clean and preprocess tweet text
  - Train a Logistic Regression model
  - Predict the sentiment of user-inputted tweets
  - Web interface for user interaction.
- **Technologies Used**:
  - Python
  - Flask
  - Scikit-learn
  - Joblib
  - HTML/CSS.
- **Data Processing**:
  The Sentiment140 dataset contains 1,600,000 tweets, each labeled as either positive or negative. The dataset includes the following columns:

  - `sentiment`: Sentiment of the tweet (0 = negative, 4 = positive)
  -  `id`: Unique identifier for the tweet
  - `date`: Date and time of the tweet
  - `query`: Query used to find the tweet (if any)
  - `user`: Username of the person who tweeted
  - `text`: Text of the tweet.
 
    To prepare the data for training, several cleaning steps were performed:

1. **Remove URLs**: Tweets often contain URLs that do not contribute to the sentiment analysis.
2. **Remove Mentions**: Mentions (e.g., @user) are removed to focus on the content of the tweet.
3. **Remove Hashtags**: Hashtags are removed to prevent skewing the sentiment analysis.
4. **Remove Special Characters**: Only alphabetic characters and spaces are retained.
5. **Convert to Lowercase**: All text is converted to lowercase to maintain consistency.
6. **Remove Stopwords**: Common English stopwords (e.g., "and", "the", "is") are removed as they do not carry significant sentiment information.

- **Model Training**:
- Choice of Model
Logistic Regression was chosen for this project due to its simplicity and effectiveness in binary classification problems. It is also computationally efficient and provides interpretable results.

  - Steps to Train the Model
   -Vectorization: Convert the cleaned text data into numerical data using CountVectorizer and TfidfTransformer.

   - CountVectorizer: Converts a collection of text documents to a matrix of token counts.
   - TfidfTransformer: Transforms the count matrix to a normalized term-frequency or TF-IDF representation.
   - Train-Test Split: Split the dataset into training and testing sets to evaluate the model's performance.

- Model Training: Train a Logistic Regression model on the training data.

- Model Evaluation: Evaluate the model's performance using accuracy, classification report, and confusion matrix..

- **Web Application**: The web application is built using Flask, which provides a simple and lightweight framework for web development in Python. The app allows users to input a tweet and get the sentiment prediction in real-time.
- twitter-sentiment-analysis/
│
├── app.py                  # Main Flask application
├── train_model.py          # Script to train and save the model
├── requirements.txt        # Project dependencies
├── templates/
│   ├── index.html          # HTML template for the home page
│   └── result.html         # HTML template for the result page
├── static/
│   └── style.css           # CSS file for styling
├── logistic_model.pkl      # Saved Logistic Regression model
├── count_vectorizer.pkl    # Saved CountVectorizer
├── tfidf_transformer.pkl   # Saved TfidfTransformer
└── training.1600000.processed.noemoticon.csv # Sentiment140 dataset

- **Usage**:
  - Accessing the Application
    - You can access the deployed application and test the model by visiting http://127.0.0.1:5000.

- Usage
  - Home Page: Enter a tweet in the text area and click "Predict".
  - Prediction Result: The application will display whether the sentiment of the tweet is "Positive" or "Negative".
  - Screenshots:
    - Home Page
      -<img width="822" alt="image" src="https://github.com/sumedhwani/Twitter-Sentiment-Analysis/assets/85736652/b061ee46-1cf0-4e1d-a115-f0d8731d6490">
    - Prediction Result
      -<img width="833" alt="image" src="https://github.com/sumedhwani/Twitter-Sentiment-Analysis/assets/85736652/ef16cda9-5b10-4ab5-bcc2-a203169a432c">
   
    
   
