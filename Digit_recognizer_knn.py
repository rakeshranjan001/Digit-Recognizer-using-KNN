import pandas as pd
import numpy as np

#importing the datasets
data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
x = data.iloc[:,1:].values
y = data.iloc[:,0].values

#splitting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Using K-NN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

y_final=classifier.predict(test)
solution = pd.DataFrame({"ImageId":test.index+1,"Label":y_final})
solution.to_csv("Digit_Recognizer_KNN.csv",index=False)
