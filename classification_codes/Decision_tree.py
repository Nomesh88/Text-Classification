#TEXT CLASSIFICATION USING LOGISTIC REGRESSION

#IMPORTING LIBRARIES  
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

#CREATING DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

#ACCURACY SCORE FOR THE MODEL
result=dtree.score(X_test,y_test)
print(result*100)
