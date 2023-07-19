from flask import Flask, request, jsonify
from joblib import load
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

clf = load('model.joblib')
vectorizer = load('vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Preprocess the text
    text = BeautifulSoup(text, 'html.parser').get_text().lower()
    text = vectorizer.transform([text])

    # Make the prediction and respond
    prediction = clf.predict(text)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)