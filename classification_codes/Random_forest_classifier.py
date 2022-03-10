#random forest classifier
# Import the model we are using
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.50, random_state=42)

#converting the text data to numerical form
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train.astype('U'))
X_train = vectorizer.transform(sentences_train.astype('U'))
X_test  = vectorizer.transform(sentences_test.astype('U'))


# Train the model on training data
filename='Text model/finalized_model[random].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train,y_train)
result = loaded_model.score(X_test, y_test)
print(result*100)
