import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

train=pd.read_csv('credit-cards.train.csv')
test=pd.read_csv('credit-cards.test.csv')
val=pd.read_csv('credit-cards.val.csv')
#df=format_data(df)
train=train.drop([0])
test=test.drop([0])
val=val.drop([0])

train_X=train.iloc[:,0:-1]
train_Y=train.iloc[:,-1]
test_X=test.iloc[:,0:-1]
test_Y=test.iloc[:,-1]
val_X=val.iloc[:,0:-1]
val_Y=val.iloc[:,-1]

#print(train_X.head())
print('using min sample split:')
min_samples=list(range(50,300,5))
min_leaf=list(range(5,300,3))
max_d=list(range(50,3000,50))
val_acc=[]
for i in min_samples:
	for j in min_leaf:
		for k in max_d:
			clf = DecisionTreeClassifier(min_samples_split=i,min_samples_leaf=j,max_depth=k)
			clf = clf.fit(train_X,train_Y)
			y_pred_test = clf.predict(test_X)
			y_pred_val=clf.predict(val_X)
			y_pred_train=clf.predict(train_X)
			print(i,j,k)
			val_acc.append((i,j,k,accuracy_score(val_Y, y_pred_val)))
			print("Accuracy of training set :",accuracy_score(train_Y, y_pred_train))
			print("Accuracy of validation set :",accuracy_score(val_Y, y_pred_val))
			print("Accuracy of test set :",accuracy_score(test_Y, y_pred_test))