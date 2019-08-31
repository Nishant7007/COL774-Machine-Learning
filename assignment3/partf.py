#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import math as math
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt


# In[33]:


train=pd.read_csv('credit-cards.train.csv')
test=pd.read_csv('credit-cards.test.csv')
val=pd.read_csv('credit-cards.val.csv')


# In[34]:


train=train.drop([0])
test=test.drop([0])
val=val.drop([0])

train_X=train.iloc[:,0:-1]
train_Y=train.iloc[:,-1]

test_X=test.iloc[:,0:-1]
test_Y=test.iloc[:,-1]

val_X=val.iloc[:,0:-1]
val_Y=val.iloc[:,-1]


# In[35]:


test_Y.shape


# In[46]:


random_forest = RandomForestClassifier(n_estimators=100,max_features=None,bootstrap=True)


# In[47]:


random_forest.fit(train_X, train_Y)


# In[48]:


test_pred = random_forest.predict(test_X)
train_pred= random_forest.predict(train_X)
val_pred= random_forest.predict(val_X)


# In[49]:


print('train accuracy',sklearn.metrics.accuracy_score(train_Y, train_pred)) 
print('test accuracy',sklearn.metrics.accuracy_score(test_Y, test_pred)) 
print('validation accuracy',sklearn.metrics.accuracy_score(val_Y, val_pred)) 


# In[ ]:





# In[ ]:




