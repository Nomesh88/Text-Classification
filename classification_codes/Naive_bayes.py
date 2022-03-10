#NAIVE BAYES

#IMPORTING THE MODELS
from sklearn.naive_bayes import GaussianNB, MultinominalNB, BernouliNB
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#DEFINING THE FEATURES AND TARGETS
df=pd.read_csv('Data3.csv')
sentences = df['word'].values
y=df['label'].values

#SPLITTING THE DATA FOR TRAINING AND TESTING USING TRAIN TEST SPLIT
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#CONVERT THE DATA INTO NUMERICAL FORM USING VECTORIZER
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))

#LOADING THE MODEL, FITTING THE TRAINING DATA AND TESTING THE MODEL (GAUSSIAN NAIVE BAYES)
filename='Text model/finalized_model[naive-bayes].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)


#LOADING THE MODEL, FITTING THE TRAINING DATA AND TESTING THE MODEL (MULTINOMIAL NAIVE BAYES CLASSIFIER)
filename='Text model/finalized_model[naive-bayes-multi].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)

#LOADING THE MODEL, FITTING THE TRAINING DATA AND TESTING THE MODEL (BERNOULI NAIVE BAYES CLASSIFIER)
filename='Text model/finalized_model[naive-bayes-ber].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)

