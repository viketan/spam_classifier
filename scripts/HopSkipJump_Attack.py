import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump
from sklearn.metrics import classification_report
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

smsdata = pd.read_csv('../data/spam.csv', encoding='Windows-1252')
smsdata.drop(smsdata.columns[2:], axis=1, inplace=True)


nltk.download(['stopwords', 'wordnet'])
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(smsdata)):
    msg = re.sub('[^a-zA-Z]', ' ', smsdata['v2'][i])
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)
    msg = [lemmatizer.lemmatize(word) for word in msg if not word in set(
        stopwords.words('english'))]
    msg = ' '.join(msg)
    corpus.append(msg)

# using Bag of word

cv = CountVectorizer(max_features=7000)
X_BOW = cv.fit_transform(corpus)

le = LabelEncoder()
smsdata['v1'] = le.fit_transform(smsdata['v1'])
Y = smsdata[['v1']]


x_train, x_test, y_train, y_test = train_test_split(
    X_BOW, Y, train_size=0.6, random_state=69)


model_endpoint = "../Artifacts/model.pkl"
model = joblib.load(model_endpoint)

score_b = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print("Accuracy before attack: ", score_b)
print(classification_report(y_test, y_pred))
art_classifier = SklearnClassifier(model=model)
# Step 5: Initialize the projected gradient descent object with ART classifier.
attack = HopSkipJump(classifier=art_classifier,
                     targeted=False, max_iter=0, max_eval=1000, init_eval=10)
# Step 6: Generate the adversial data with test data.
X_train = x_train.A
x_test_adv = attack.generate(X_train)
# Step 7: Compute the score.
score = model.score(x_test_adv, y_train)
y_pred = model.predict(x_test_adv)
print(classification_report(y_train, y_pred))
art_classifier = SklearnClassifier(model=model)
print("Accuracy after attack: ", score)
data = np.concatenate([X_train, x_test_adv])
y_data = np.concatenate([y_train, y_train])
data = np.concatenate([data, y_data], axis=1)
new = pd.DataFrame(data=data)
new.to_csv("../data/Adv_data.csv", sep=',', encoding='utf-8', index=False)
