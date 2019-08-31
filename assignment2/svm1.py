import sys
train_file=str(sys.argv[1])
test_file=str(sys.argv[2])
mul_or_bin=str(sys.argv[3])
part_num=str(sys.argv[4])

import sys
import numpy as np
import pandas as pd
from cvxopt import matrix as cvx_matrix, solvers
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform as sqf
from numpy.linalg import norm
#import itertools
#from libsvm import *

train=pd.read_csv(train_file,header=None)
test=pd.read_csv(train_file,header=None)

def binary_part_a(train,test):
    train=train.loc[(train[784] == 2)|(train[784] == 3)]
    train=train.reset_index(drop=True)
    train_X=train.iloc[:,0:784]
    train_Y=train.iloc[:,784]
    train_X=(train_X.as_matrix())/255
    train_Y=train_Y.as_matrix()
    train_Y[train_Y==2]=-1
    train_Y[train_Y==3]=1
    test=test.loc[(test[784] == 2)|(test[784] == 3)]
    test=test.reset_index(drop=True)
    test_X=test.iloc[:,0:784]
    test_Y=test.iloc[:,784]
    test_X=(test_X.as_matrix())/255
    test_Y=test_Y.as_matrix()
    test_Y[test_Y==2]=-1
    test_Y[test_Y==3]=1
    H=train_Y[:,None]*train_X
    K=H@np.transpose(H)
    pts=train_X.shape[0]
    f=train_X.shape[1]
    P=cvx_matrix(K)
    q=-1*cvx_matrix(np.ones((pts,1)))
    m=np.diag(-1*np.ones(pts))
    n=np.identity(pts)
    G=cvx_matrix(np.vstack((m,n)))
    h=cvx_matrix(np.hstack((np.zeros(pts),np.ones(pts))))
    A=cvx_matrix(train_Y.reshape(1,-1).astype(np.double))
    b=cvx_matrix(np.zeros(1))
    result=solvers.qp(P,q,G,h,A,b)
    x_val=np.array(result['x'])
    w=np.sum(x_val*train_Y[:,None]*train_X,axis=0)
    b=(train_Y[(x_val>0.0001).reshape(-1)]- (train_X[(x_val>0.0001).reshape(-1)].dot(w)))[0]
    pts=test_Y.shape[0]
    pred_y=test_X.dot(w)+b
    pred_y[pred_y<0]=-1
    pred_y[pred_y>=0]=1
    accuracy=np.sum(pred_y==test_Y)/pts*100
    print(accuracy)


def binary_part_b(train,test):
    train=train.loc[(train[784] == 2)|(train[784] == 3)]
    train=train.reset_index(drop=True)
    train_X=train.iloc[:,0:784]
    train_Y=train.iloc[:,784]
    train_X=(train_X.as_matrix())/255
    train_Y=train_Y.as_matrix()
    train_Y[train_Y==2]=-1
    train_Y[train_Y==3]=1
    test=test.loc[(test[784] == 2)|(test[784] == 3)]
    test=test.reset_index(drop=True)
    test_X=test.iloc[:,0:784]
    test_Y=test.iloc[:,784]
    test_X=(test_X.as_matrix())/255
    test_Y=test_Y.as_matrix()
    test_Y[test_Y==2]=-1
    test_Y[test_Y==3]=1
    R=pdist(train_X,'sqeuclidean')
    K=np.exp(-0.05*sqf(R))
    x=train_Y[:,None]
    y=np.transpose(train_Y[:,None])
    K=x@y*K
    pts=train_X.shape[0]
    p=cvx_matrix(K)
    q=cvx_matrix(-1*np.ones((pts,1)))
    m=np.diag(-1*np.ones(pts))
    n=np.identity(pts)
    G=cvx_matrix(np.vstack((m,n)))
    h=cvx_matrix(np.hstack((np.zeros(pts),np.ones(pts))))
    A=cvx_matrix(train_Y.reshape(1,-1).astype(np.double))
    b=cvx_matrix(np.zeros(1))
    result=solvers.qp(p,q,G,h,A,b)
    x_val=np.array(result['x']).flatten()
    val=.0001
    y_val=np.arange(len(x_val))[x_val>val]
    alpha=x_val[x_val>val]
    xi=train_X[x_val>val]
    yi=train_Y[x_val>val]
    b=0
    for i in range(len(yi)):
        b+=yi[i]
        temp=K[y_val[i],x_val>val]
        b-=np.sum(alpha * temp * yi)
    b/=len(yi) 
    pts=test_X.shape[0]
    pred_y=np.zeros(pts)
    for i in range(pts):
        val=0
        for j in range(len(alpha)):
            temp=test_X[i]-xi[j]
            temp=np.exp((temp.dot(temp))*0.05*-1)
            val+=alpha[j]*yi[j]*temp
        pred_y[i]=val  
    pred_y+=b  
    pred_y[pred_y>=0]=1
    pred_y[pred_y<0]=-1
    accuracy=(np.sum(pred_y==test_Y))/pts*100
    print(accuracy)

def train_data_format(data,val1,val2):
    data=data.loc[(data[784] == val1) | (data[784] == val2)]
    data=data.reset_index(drop=True)
    data_X=((data.iloc[:,0:784]).as_matrix())/255
    data_Y=((data.iloc[:,784]).as_matrix()).astype(float)
    data_Y[data_Y==val1]=-1
    data_Y[data_Y==val2]=1
    return data_X,data_Y    

def test_data_format(data,val1,val2):
    #data=data.loc[(data[784] == val1) | (data[784] == val2)]
    data=data.reset_index(drop=True)
    data_X=((data.iloc[:,0:784]).as_matrix())/255
    data_Y=((data.iloc[:,784]).as_matrix()).astype(float)
    data_Y[data_Y==val1]=-1
    data_Y[data_Y==val2]=1
    return data_X,data_Y  

def multi_part_a(train,test):
    k=list(itertools.combinations([i for i in range(5)],2))
    #k=[(0,1),(7,9)]
    total_classes=len(k)
    final_pred=[]
    list3=[]
    for i in range(total_classes):
        train_X,train_Y=train_data_format(train,k[i][0],k[i][1])
        test_X,test_Y=test_data_format(test,k[i][0],k[i][1])
        R=pdist(train_X,'sqeuclidean')
        K=np.exp(-0.05*sqf(R))
        x=train_Y[:,None]
        y=np.transpose(train_Y[:,None])
        K=x@y*K
        pts=train_X.shape[0]
        p=cvx_matrix(K)
        q=cvx_matrix(-1*np.ones((pts,1)))
        m=np.diag(-1*np.ones(pts))
        n=np.identity(pts)
        G=cvx_matrix(np.vstack((m,n)))
        h=cvx_matrix(np.hstack((np.zeros(pts),np.ones(pts))))
        A=cvx_matrix(train_Y.reshape(1,-1).astype(np.double))
        b=cvx_matrix(np.zeros(1))
        result=solvers.qp(p,q,G,h,A,b)
        x_val=np.array(result['x']).flatten()
        #print(x_val)
        val=.0001
        y_val=np.arange(len(x_val))[x_val>val]
        alpha=x_val[x_val>val]
        #print(alpha)
        xi=train_X[x_val>val]
        #print(xi)
        yi=train_Y[x_val>val]
        #print(yi)
        b=0
        for ii in range(len(yi)):
            b+=yi[ii]
            temp=K[y_val[ii],x_val>val]
            b-=np.sum(alpha * temp * yi)
        b/=len(yi)
        pts=test_X.shape[0]
        pred_y=np.zeros(pts)
        for iii in range(pts):
            val=0
            for j in range(len(alpha)):
                temp=test_X[iii]-xi[j]
                temp=np.exp((temp.dot(temp))*0.05*-1)
                val+=alpha[j]*yi[j]*temp
            pred_y[iii]=val  
        pred_y+=b
        pred_y[pred_y>=0]=1
        pred_y[pred_y<0]=-1
        accuracy=(np.sum(pred_y==test_Y))/pts*100
        print(accuracy)
        pred_y[pred_y<0]=k[i][0]
        pred_y[pred_y>0]=k[i][1]
        final_pred.append(pred_y)
        list2=[]
        for j in final_pred[i]:
            if j==1:
                list2.append(k[i][1])
            else:
                list2.append(k[i][0])
        list3.append(list2) 
        print('value of i is:'+str(i))
    list4=np.transpose(np.array(list3))
    final_predcition=[]
    for it in range(list4.shape[0]):
        val=max(set(list(list4[i])), key=list(list4[i]).count)
        final_predcition.append(val)
    final_predcition=np.array(final_predcition)
    accuracy=(np.sum(final_predcition==test_Y))/pts*100
    print(accuracy)    
    return final_predcition


def part_c_linear(test,train):
	train=train.loc[(train.iloc[:,784] == 2) | (train.iloc[:,784] == 3)]
	train=train.reset_index(drop=True)
	train_X=train.iloc[:,0:784]
	train_Y=train.iloc[:,784]
	train_X=(train_X.as_matrix())/255
	train_Y=train_Y.as_matrix()
	train_Y[train_Y==2]=-1
	train_Y[train_Y==3]=1


	test=test.loc[(test.iloc[:,784] == 2) | (test.iloc[:,784] == 3)]
	test=test.reset_index(drop=True)
	test_X=test.iloc[:,0:784]
	test_Y=test.iloc[:,784]
	test_X=(test_X.as_matrix())/255
	test_Y=test_Y.as_matrix()
	test_Y[test_Y==2]=-1
	test_Y[test_Y==3]=1

	x = time()
	prob = svm_problem(train_Y,train_X)
	param = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
	m = svm_train(prob, param)
	y = time()
	print(y-x)
	p_label, p_acc, p_val = svm_predict(test_Y, test_X, m)
	print(p_acc)


def part_c_gaussian(test,train):
	train=train.loc[(train.iloc[:,784] == 2) | (train.iloc[:,784] == 3)]
	train=train.reset_index(drop=True)
	train_X=train.iloc[:,0:784]
	train_Y=train.iloc[:,784]
	train_X=(train_X.as_matrix())/255
	train_Y=train_Y.as_matrix()
	train_Y[train_Y==2]=-1
	train_Y[train_Y==3]=1


	test=test.loc[(test.iloc[:,784] == 2) | (test.iloc[:,784] == 3)]
	test=test.reset_index(drop=True)
	test_X=test.iloc[:,0:784]
	test_Y=test.iloc[:,784]
	test_X=(test_X.as_matrix())/255
	test_Y=test_Y.as_matrix()
	test_Y[test_Y==2]=-1
	test_Y[test_Y==3]=1

	x = time()
	prob = svm_problem(train_Y,train_X)
	param = svm_parameter('-s 0 -t 0 -c 1 -g 0.05')
	m = svm_train(prob, param)
	y = time()
	print(y-x)
	p_label, p_acc, p_val = svm_predict(test_Y, test_X, m)
	print(p_acc)

def binary_part_c(test,train):	
	part_c_linear(test,train)
	part_c_gaussian(test,train)

def multiclass_format(data):
    train_X = data.iloc[:,0:-1]
    train_X = train_X.as_matrix()
    train_Y = data.iloc[:,-1]
    train_Y = train_Y.as_matrix()
    train_X=train_X.astype(float)
    train_Y=train_Y.astype(float)
    return train_X,train_Y

def mul_part_b(train,test):
	train_X,train_Y =multiclass_format(train)
	test_X,test_Y=multiclass_format(test)
	x = time()
	prob = svm_problem(train_Y,train_X/255)
	param = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
	m = svm_train(prob, param)
	y = time()
	print(y-x)
	p_label, p_acc, p_val = svm_predict(test_Y, test_X/255, m)
	print(p_acc)    

def validation_format(train,test):
    data = shuffle(data)
    train_X = data.iloc[0:18000,0:-1]
    train_Y = data.iloc[0:18000,-1]
    val_X = data.iloc[18000:20000,0:-1]
    val_Y = data.iloc[18000:20000,-1]
    train_X = train_X.as_matrix()
    train_Y = train_Y.as_matrix()
    train_X=train_X.astype(float)
    train_Y=train_Y.astype(float)
    return train_X,train_Y,val_X,val_Y

def multi_part_d(train,test):
    print('done')
    train_X,train_Y,val_X,val_Y =validation_format(train)
    test_X,test_Y=multiclass_format(test)
    C = []
    val_acc = []
    tst_acc = []

    prob = svm_problem(train_Y,train_X/255)
    for x in (1e-5,1e-4 ,1,5,10):
        ''' For validation test accuracy '''
        param = svm_parameter('-s 0 -t 2 -c {} -g 0.05'.format(x))  ## Changing the slack variable
        m = svm_train(prob,param)
        _,valacc,_ = svm_predict(val_Y,val_X/255,m)
        _,tstacc,_ = svm_predict(test_Y,test_X/255,m)
        C.append(x)
        val_acc.append(valacc[0])
        tst_acc.append(tstacc[0])

    x = np.log(C)
    y = val_acc
    z = tst_acc
    plt.plot(x,y,label = 'Validation Set')
    plt.plot(x,z,label = 'Test Set')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Set Accuracy')
    plt.legend()
    plt.show()


if(mul_or_bin=="1"):
	if(part_num=="a"):
		binary_part_a(train,test)
	elif(part_num=="b"):
		binary_part_b(train,test)
	elif(part_num=="c"):
		binary_part_c()
else:
	if(part_num=="a"):
		binary_part_a()
	elif(part_num=="b"):
		mul_part_b()
	elif(part_num=="d"):
		binary_part_d()


		
		