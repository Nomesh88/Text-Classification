from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
X, y = df.word.fillna(' '), df.label
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100) 

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', NearestCentroid()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))
