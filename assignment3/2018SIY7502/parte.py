import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree 


train=pd.read_csv('credit-cards.train.csv')
test=pd.read_csv('credit-cards.test.csv')
train=train.drop([0])
test=test.drop([0])
train = pd.get_dummies(train,columns=['X3','X4','X5','X6','X7','X8','X9','X10','X11'])
test = pd.get_dummies(test,columns=['X3','X4','X5','X6','X7','X8','X9','X10','X11'])
train_X=train.iloc[:,0:-1]
train_Y=train.iloc[:,-1]
test_X=test.iloc[:,0:-1]
test_Y=test.iloc[:,-1]
main_list1 = list(np.setdiff1d(list(train_X.columns),list(test_X.columns)))
#print(main_list1)
for each in main_list1:
	test_X[each]='0'
main_list2 = np.setdiff1d(list(test_X.columns),list(train_X.columns))
for each in main_list2:
	train_X[each]='0'
#print(train_X.shape,test_X.shape)
clf = DecisionTreeClassifier(min_samples_split=100,min_samples_leaf=5,max_depth=10)
clf = clf.fit(train_X,train_Y)
y_pred_test = clf.predict(test_X)
y_pred_train=clf.predict(train_X)
print("Accuracy of training set :",accuracy_score(train_Y, y_pred_train)*100.0)
print("Accuracy of test set :",accuracy_score(test_Y, y_pred_test)*100.0)


train=pd.read_csv('credit-cards.train.csv')
val=pd.read_csv('credit-cards.val.csv')
train=train.drop([0])
val=val.drop([0])
train = pd.get_dummies(train,columns=['X3','X4','X5','X6','X7','X8','X9','X10','X11'])
val = pd.get_dummies(val,columns=['X3','X4','X5','X6','X7','X8','X9','X10','X11'])
train_X=train.iloc[:,0:-1]
train_Y=train.iloc[:,-1]
val_X=val.iloc[:,0:-1]
val_Y=val.iloc[:,-1]
main_list1 = list(np.setdiff1d(list(train_X.columns),list(val_X.columns)))
#print(main_list1)
for each in main_list1:
	val_X[each]='0'
main_list2 = np.setdiff1d(list(val_X.columns),list(train_X.columns))
for each in main_list2:
	train_X[each]='0'
#print(train_X.shape,val_X.shape)
clf = DecisionTreeClassifier(min_samples_split=100,min_samples_leaf=5,max_depth=10)
clf = clf.fit(train_X,train_Y)
y_pred_val = clf.predict(val_X)
y_pred_train=clf.predict(train_X)
print("Accuracy of validation set :",accuracy_score(val_Y, y_pred_val)*100.0)