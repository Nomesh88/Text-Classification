#RANDOM FOREST CLASSIFIER

# IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
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


#LOAD THE MODEL AND FIT 
filename='Text model/finalized_model[random].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)

#PRINT THE ACCURACY SCORE OF THE MODEL
result = loaded_model.score(X_test, y_test)
print(result*100)
