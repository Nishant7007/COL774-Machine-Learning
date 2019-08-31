
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[27]:


def cf(a,y):
    return 0.5*((np.linalg.norm(a-y)**2)/len(a))

def rdf(x):
    return 1*(x>0)

def sf(x):
    return 1/(1+np.exp(-x))

def rf(x):
    return x*(x>0)

def sdf(x):
    return x*(1-x)

def back_prop(op, yb, w, relu=False):
    db = []
    value = len(w)+1
    dw = []
    for i in range(1, value):
        if i!=1 and relu:
            gr = rdf(op[-i])
            lst1=[]
        else:
            gr = sdf(op[-i])
            lst1=[]
        if(i==1):
            delta = gr*(op[-1] - yb)
            lst1=[]
        else:
            delta = gr*(w[-i + 1].T.dot(delta))
            lst1=[]         
        db.append(delta)
        dw.append(delta.dot(op[-i-1].T))
    lst1=[]
    db.reverse()
    lst2=[]
    dw.reverse()
    return dw,db

def forward_prop(op_i, w, b, relu=False):
    op = []
    n = len(w)+1
    op.append(op_i)
    lst1 = []
    for i in range(n-1):
        op_i = np.dot(w[i], op_i) + b[i]
        if relu and i < n - 2:
            op_i = rf(op_i)
            lst1=[]
        else:
            op_i = sf(op_i)
            lst1=[]
        op.append(op_i)
    return op


# In[28]:




def train_NN(X, Y, lrate, batch_size, epochs, w, b, relu=False, dlrate=False, err_tol=10**-4):
    err = np.inf
    err_trhld=10**-6
    err_1 = np.inf
    index = np.arange(len(X))
    err_2 = np.inf  
    for eph in range(epochs):
        np.random.shuffle(index)
        print('epoch : ',eph)
        if err_tol>(err_1 - err_2) and dlrate and err_tol>(err - err_1):
            lrate /= 0.2
        for i in range(0, len(X), batch_size):
            lst = []
            batch = index[i : i+batch_size]
            lst = []
            bX, bY = X[batch], Y[batch]
            for bx, by in zip(bX, bY):
                op = forward_prop(bx.reshape([-1,1]), w, b, relu)
                dw, db = back_prop(op, by.reshape([-1,1]), w, relu)
                w_delta = []
                lst = []
                b_delta = []
                for j in range(len(w)):
                    val = w[j].shape
                    w_delta.append(np.zeros(val))
                    lst = []
                    val = b[j].shape
                    b_delta.append(np.zeros(val))                    
                for j,(dwv, dbv) in enumerate(zip(dw,db)):
                    w_delta[j] += dwv
                    lst = []
                    b_delta[j] += dbv
                    lst = []
            for k in range(len(w)):
                w[k] -= w_delta[k] * lrate
                lst = []
                b[k] -= b_delta[k] * lrate
                lst = []
        err_2 = err_1
        err_1 = err
        err = cf(forward_prop(X.T, w, b, relu)[-1].T, Y)
        print('Error : ',err)
    return w,b


# In[32]:


def final_prediction(results):
    l=[]
    for i in results:
        val = max(i)
        temp = val==i
        l.append(temp.astype(int).tolist())
        lst=[]
    return l

def get_accuracy_cmatrix(l1,l2):
    count=0
    for a,b in zip(l1,l2):
        if a==b:
            count = count+1
    acc = count/len(l1)
    lst1 = []
    val = len(l1)
    lst2 = []
    for i in range(val):
        lst1.append(l1[i].index(1))
        lst2.append(l2[i].index(1))
    col = [0,1,2,3,4,5,6,7,8,9]
    cmatrix = confusion_matrix(lst1,lst2,col)
    return acc, cmatrix

def Built_and_train_NN(data_X, data_Y, nperceptrons, lrate=0.1, batch_size=100, epochs=10, dlrate=False, relu=False):
    weights = []
    bias = []
    nlayers = len(nperceptrons)
    for i in range(nlayers-1):
        weights.append(np.random.randn(nperceptrons[i+1], nperceptrons[i]) / np.sqrt(i+1))
        bias.append(np.random.randn(nperceptrons[i+1],1)) 
    weights, bias = train_NN(data_X, data_Y, lrate, batch_size, epochs, weights, bias, relu, dlrate)
    return weights, bias
    
def test_NN(test_data_X, test_data_Y, weights, bias, relu=False):
    results = forward_prop(test_data_X.T, weights, bias, relu)[-1].T
    return get_accuracy_cmatrix(train_data_Y.astype(int).tolist(), np.array(final_prediction(results)).tolist())


# In[ ]:


'''**************************** PART-A **************************************'''


# In[ ]:


def onehotencodecol(a):
    if np.min(a)!=0:
        a=a-np.min(a)
    b = np.zeros((len(a), np.max(a)+1))
    b[np.arange(len(a)), a] = 1
    return b

def onehotencode(data):
    m,n=np.shape(data)
    s=np.zeros(m)
    for i in range(n):
        s=np.vstack((s,onehotencodecol(data[:,i]).T))
    return s[1:,:].astype('int').T


# In[38]:


data = pd.read_csv('poker-hand-training-true.data',header = None)
data1 = onehotencode(np.array(data))
train_data_X = data1.T[:85].T
train_data_Y = data1.T[85:].T

data = pd.read_csv('poker-hand-testing.data',header = None)
data1 = onehotencode(np.array(data))
test_data_X = data1.T[:85].T
test_data_Y = data1.T[85:].T


# In[ ]:


'''**************************** PART-C **************************************'''


# In[34]:


def partc(train_data_X, train_data_Y):
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,10])
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
partc(train_data_X, train_data_Y)


# In[ ]:


'''**************************** PART-D **************************************'''


# In[35]:


def partd(train_data_X, train_data_Y):
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,5*i,10])
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
partd(train_data_X, train_data_Y)


# In[ ]:


'''**************************** PART-E **************************************'''


# In[36]:


def parte(train_data_X, train_data_Y):
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,10], dlrate=True)
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,5*i,10], dlrate=True)
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
    
parte(train_data_X, train_data_Y)


# In[ ]:


'''**************************** PART-F **************************************'''


# In[37]:


def partf(train_data_X, train_data_Y):
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,10], relu=True)
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
    for i in range(1,6):
        t1 = time.time()
        weights, bias = Built_and_train_NN(train_data_X, train_data_Y, [85,5*i,5*i,10], relu=True)
        t2 = time.time()
        print(t2-t1)
        print(test_NN(train_data_X, train_data_Y, weights, bias))
        print(test_NN(test_data_X, test_data_Y, weights, bias))
    
partf(train_data_X, train_data_Y)

