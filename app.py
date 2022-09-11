import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
import numpy as np
from flask import Flask, request, render_template
from uuid import uuid4
# Called when the service is loaded


def init():
    global model
    # Get the path to the deployed model file and load it
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'spam_model.pkl')
    model = joblib.load('../Artifacts/model.pkl')


def preprocess(raw_data):
    lemmatizer = WordNetLemmatizer()
    msg = re.sub('[^a-zA-Z]', ' ', raw_data)
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)
    msg = [lemmatizer.lemmatize(word) for word in msg if not word in set(
        stopwords.words('english'))]
    msg = ' '.join(msg)
    corpus = pd.read_csv('../data/corpus.csv', header=None)
    corpus = corpus.values
    corpus = corpus.flatten()
    corpus = corpus.tolist()
    corpus.append(msg)
    cv = CountVectorizer(max_features=7000)
    X_BOW = cv.fit_transform(corpus)
    return X_BOW

# Called when a request is received


def run(raw_data):
    #raw_data = json.loads(raw_data)
    # Get a prediction from the model
    preprocessed_data = preprocess(raw_data)
    predictions = model.predict(preprocessed_data[-1])
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['NOT spam', 'spam']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)


''''
init()
msg = input("Enter SMS: ")
raw_data = json.dumps({"data": msg})
predictions = run(raw_data)
predicted_classes = json.loads(predictions)
print(predicted_classes[0])
'''

app = Flask(__name__)

node_addr = str(uuid4()).replace('-', '')


@app.route('/')
def home():
    return render_template('home.html')
   # return '<html><body><h1>Hello World</h1></body></html>'


@app.route('/predict', methods=['POST'])
def predict():
    sms = [x for x in request.form.values()]
    sms = sms[0]
    '''
    sms = request.get_json()
    keys = ['data']
    if not all(key in sms for key in keys):
        return 'some elements are missing', 400
    '''
    init()
    predictions = run(sms)
    predicted_classes = json.loads(predictions)
    response = {'message': "This SMS is "+predicted_classes[0]}
    return render_template('home.html', prediction_text=response['message'])


app.run(host='0.0.0.0', port=5123, debug=True)
