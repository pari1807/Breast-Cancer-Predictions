import numpy as np  
import pandas as pd 
import sklearn 
import warnings
warnings.filterwarnings("ignore")

dataset=pd.read_csv('data.csv')
print(dataset)
#how many columns are there 
dataset.info()
#data cleaning
dataset.drop(dataset.columns[[-1, 0]],axis=1,inplace=True)
#how many have b and m type of diagnosis
print(dataset['diagnosis'].value_counts())
#feature scaling
from  sklearn.model_selection import train_test_split
diag_map = {'M' : 1, 'B' : 0}
dataset['diagnosis']=dataset['diagnosis'].map(diag_map)
print(dataset)
X=dataset[['radius_mean','perimeter_mean','area_mean','concavity_mean','concave points_mean']]
print(X)
#train 
new_dataset=dataset.drop(dataset.columns[[0]],axis=1,inplace=True)
print(new_dataset)
y=dataset[['diagnosis']]
X_train,y_train,X_test,y_test=train_test_split(dataset,y,test_size=0.2,random_state=42)
#model-KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_y_pred=knn.pred(X_test)
#accuracy check
from sklearn.metrics import accuracy_score
accuracy_score(knn_y_pred,y_test)

#model-Logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
print(lr.fit(X_train,y_train))
lr_y_pred =lr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(lr_y_pred,y_test)

#model-Naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
gnb_y_pred=gnb.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(gnb_y_pred,y_test)

#model-K fold-cross validation
#it will take the average accuracy,everytime you shuffle the dataset the model will give another accuracy 
from sklearn.model_selection import cross_val_score
accuracy_all=[]
cvs_all=[]
scores=cross_val_score(knn,X,y,cv=10)
accuracy_all.append(accuracy_score(knn_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("Accuracy: {0:.2%}".format(accuracy_score(knn_y_pred,y_test)))
print("10-fold Cross validation score: {0:.2%(+/- {1:.2%})}".format(np.mean(scores),np.std(scores)*2))

accuracy_all=[]
cvs_all=[]
scores=cross_val_score(lr,X,y,cv=10)
accuracy_all.append(accuracy_score(lr_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("Accuracy: {0:.2%}".format(accuracy_score(lr_y_pred,y_test)))
print("10-fold Cross validation score: {0:.2%(+/- {1:.2%})}".format(np.mean(scores),np.std(scores)*2))

accuracy_all=[]
cvs_all=[]
scores=cross_val_score(gnb,X,y,cv=10)
accuracy_all.append(accuracy_score(gnb_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("Accuracy: {0:.2%}".format(accuracy_score(gnb_y_pred,y_test)))
print("10-fold Cross validation score: {0:.2%(+/- {1:.2%})}".format(np.mean(scores),np.std(scores)*2))
