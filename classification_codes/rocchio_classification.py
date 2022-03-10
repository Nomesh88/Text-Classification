#ROCCHIO CLASSIFICATION

# IMPORTING THE LIBRARIES
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

#DEFINING THE FEATURES AND TARGETS
df=pd.read_csv('Data3.csv')
sentences = df['word'].values
y=df['label'].values

#SPLITTING THE DATA FOR TRAINING AND TESTING USING TRAIN TEST SPLIT
X, y = df.word.fillna(' '), df.label
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100) 

#DEFINING A PIPELINE FOR THE MODEL
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', NearestCentroid()),
                     ])

text_clf.fit(X_train, y_train)

#TESTING THE MODEL
predicted = text_clf.predict(X_test)

#PRINTING THE CLASSIFICAITON REPORT FOR THE MODEL
print(metrics.classification_report(y_test, predicted))
