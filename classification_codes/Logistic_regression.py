import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
df=pd.read_csv('Data3.csv')
sentences = df['word'].values
y=df['label'].values

#training and testing values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#converting the text data to numerical form
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))

#specifying the model and fitting the data
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)
