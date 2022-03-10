# IMPORTING THE LIBRARIES
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#DEFINING THE FEATURES AND TARGETS
df=pd.read_csv('Text model/dataset2/Data3.csv')
sentences = df['word'].values
y=df['label'].values


#SPLITTING THE DATA FOR TRAINING AND TESTING USING TRAIN TEST SPLIT
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#CONVERT THE DATA INTO NUMERICAL FORM USING VECTORIZER
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))

#DEFINING THE MODEL
filename='Text model/finalized_model[knn].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)

#ACCURACY OF THE MODEL
result = loaded_model.score(X_test, y_test)
print(result*100)
