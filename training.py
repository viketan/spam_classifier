import pandas as pd
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
warnings.filterwarnings("ignore")


smsdata = pd.read_csv('../data/spam.csv', encoding='Windows-1252')
smsdata.drop(smsdata.columns[2:], axis=1, inplace=True)


#nltk.download(['stopwords', 'wordnet'])
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

np.savetxt("../data/corpus.csv", corpus, delimiter=", ", fmt='% s')
# using Bag of word

cv = CountVectorizer(max_features=7000)
X_BOW = cv.fit_transform(corpus)

le = LabelEncoder()
smsdata['v1'] = le.fit_transform(smsdata['v1'])
Y = smsdata[['v1']]


x_train, x_test, y_train, y_test = train_test_split(
    X_BOW, Y, train_size=0.7, random_state=69)


multinb = MultinomialNB()
model_nb = multinb.fit(x_train, y_train)

# prediction
y_pred = model_nb.predict(x_test)


print(accuracy_score(y_pred, y_test))


# create an iterator object with write permission - model.pkl
with open('../Artifacts/model.pkl', 'wb') as files:
    pickle.dump(model_nb, files)
