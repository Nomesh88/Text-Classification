#Naive bayes
from sklearn.naive_bayes import GaussianNB
filename='Text model/finalized_model[naive-bayes].sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.fit(X_train.todense(),y_train)
result1=loaded_model.score(X_test.todense(),y_test)
result = loaded_model.predict(X_test.todense())
print(result1*100)
