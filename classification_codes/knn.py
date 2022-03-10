import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv('Text model/dataset2/Data3.csv')
sentences = df['word'].values
y=df['label'].values
#training and testing values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#converting the text data to numerical form
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))

#loading the model
filename='Text model/finalized_model[knn].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)
result = loaded_model.score(X_test, y_test)
print(result*100)
