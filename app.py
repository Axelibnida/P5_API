{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6567643e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from joblib import load\n",
    "from bs4 import BeautifulSoup\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "clf = load('model.joblib')\n",
    "vectorizer = load('vectorizer.joblib')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    data = request.get_json(force=True)\n",
    "    text = data['text']\n",
    "\n",
    "    # Preprocess the text\n",
    "    text = BeautifulSoup(text, 'html.parser').get_text().lower()\n",
    "    text = vectorizer.transform([text])\n",
    "\n",
    "    # Make the prediction and respond\n",
    "    prediction = clf.predict(text)\n",
    "    return jsonify({'prediction': prediction[0]})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
