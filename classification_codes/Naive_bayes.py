#NAIVE BAYES


# IMPORTING THE LIBRARIES
from sklearn.naive_bayes import GaussianNB, MultinominalNB, BernouliNB
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#LOADING THE DATASET AND DEFINING THE FEATURES AND TARGETS
df=pd.read_csv('Data3.csv')
sentences = df['word'].values
y=df['label'].values

#SPLITTING THE DATASET WITH TRAIN_TEST_SPLIT
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#TEXT DATA TO NUMERICAL FORM USING VECTORIZER
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))

#LOADING THE MODEL FILE (GAUSSIAN NAIVE BAYES)
filename='Text model/finalized_model[naive-bayes].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)


#LOADING THE MODEL FILE (MultinominalNB Classifier)
filename='Text model/finalized_model[naive-bayes-multi].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)

#LOADING THE MODEL FILE (BernouliNB Classifier)
filename='Text model/finalized_model[naive-bayes-ber].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)

