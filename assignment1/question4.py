#QUESTION 4
#IMPORT ALL LIBRARIES

import sys

input_file_path=sys.argv[1]
output_file_path=sys.argv[2]
which_fun=int(sys.argv[3])

import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
from matplotlib import animation
from numpy.linalg import pinv
import math
from statistics import mean

#LOADING DATA AND PREPROCESSING

X=np.loadtxt(input_file_path)
Y=np.loadtxt(output_file_path,dtype=object)
X=(X-X.mean(axis=0))/X.std(axis=0)
Y=np.array([0 if i=='Alaska' else 1 for i in Y]).reshape((100,1))

def partabc(X,Y):
    print('linear discriminat analysis')
    one_list=[]
    zero_list=[]
    c_one=0
    c_zero=0
    for i in range(100):
        if Y[i]==0:
            c_zero+=1
            zero_list.append(X[i])
        else:
            c_one+=1
            one_list.append(X[i])
    mean1=np.mean(one_list,axis=0)
    mean0=np.mean(zero_list,axis=0)        
    var=np.concatenate((one_list-mean1,zero_list-mean0))
    covariance_matrix=np.transpose(var).dot(var)/100
    #covariance_matrix        

    #calculating mean and covariance matrix
    mean1=np.mean(one_list,axis=0)
    mean0=np.mean(zero_list,axis=0)
    b=(-1/2)*(np.transpose(mean0+mean1).dot(pinv(covariance_matrix).dot(mean0-mean1)))
    a0,a1=pinv(covariance_matrix).dot(mean0-mean1)
    colors=[0 if i==0 else 1 for i  in Y]
    plt.title('linear discriminant analysis')
    plt.scatter(X[0:50,0], X[0:50,1],color ='g')
    plt.scatter(X[50:,0], X[50:,1],color = 'b')
    plt.legend(('Alaska','Canada'))
    X1 = (b - a0 * X[:,0]) / (a1)
    plt.plot(X[:,0],X1, "r", label='Decision Boundary')
    plt.xlabel('Ring Diameter in fresh water')
    plt.ylabel('Ring Diameter in fresh water')
    print('mean of first class is:'+str(mean0))
    print('mean of second class is:'+str(mean1))
    print('covariance matrix is:'+str(covariance_matrix))
    plt.savefig('linear.jpg')
    plt.show()

def partdef(X,Y):
    print('GAUSSIAN DISCRIMINANT ANALYSIS')
    one_list=[]
    zero_list=[]
    c_one=0
    c_zero=0
    for i in range(100):
        if Y[i]==0:
            c_zero+=1
            zero_list.append(X[i])
        else:
            c_one+=1
            one_list.append(X[i])
    mean1=np.mean(one_list,axis=0)
    mean0=np.mean(zero_list,axis=0)
    var0=np.array(zero_list-mean0)
    var1=np.array(one_list-mean1)
    covariance_matrix0=np.cov(X[0:50,:].T)
    covariance_matrix1=np.cov(X[50:,:].T)
    print('mean of first class is:'+str(mean0))
    print('mean of second class is:'+str(mean1))
    print('covariance matrix of first class is:'+str(covariance_matrix0))
    print('covariance matrix of second class is:'+str(covariance_matrix1))
    inv0=pinv(covariance_matrix0)
    inv1=pinv(covariance_matrix1)
    cov_diff=inv1-inv0
    det0=np.linalg.det(covariance_matrix0)
    det1=np.linalg.det(covariance_matrix1)
    A=inv0-inv1
    B=-2*(np.transpose(mean0).dot(inv0)-np.transpose(mean1).dot(inv1))
    C=np.transpose(mean0).dot(inv0).dot(mean0)-np.transpose(mean1).dot(inv1).dot(mean1)-2*np.log((0.5/0.5)*(det1/det0))
    x1,x2=np.linspace(X[:,0].min()-1,X[:,0].max(),100),np.linspace(X[:,1].min()-1,X[:,1].max(),100)
    x11,x22=np.meshgrid(x1,x2)
    xx=np.c_[x11.flatten(),x22.flatten()]
    def funcv(A,B,C,x):
    	return x.dot(A).dot(x.T)+B.dot(np.transpose(x))+C
    z=np.empty((xx.shape[0],1))
    i=0
    for x in xx:
    	res=funcv(A,B,C,x)
    	z[i]=res
    	i+=1
    z=z.reshape(x11.shape)



    var=np.concatenate((one_list-mean1,zero_list-mean0))
    covariance_matrix=np.transpose(var).dot(var)/100

    b=(-1/2)*(np.transpose(mean0+mean1).dot(pinv(covariance_matrix).dot(mean0-mean1)))
    a0,a1=pinv(covariance_matrix).dot(mean0-mean1)
    colors=[0 if i==0 else 1 for i  in Y]
    fig,ax=plt.subplots()
    plt.title('linear discriminant analysis')
    #plt.scatter(X[0:50,0], X[0:50,1],color ='g')
    #plt.scatter(X[50:,0], X[50:,1],color = 'b')
    plt.legend(('Alaska','Canada'))
    X1 = (b - a0 * X[:,0]) / (a1)
    ax.plot(X[:,0],X1, "r", label='Decision Boundary')





    #fig,ax=plt.subplots()
    ax.contour(x11,x22,z,[0])
    ax.scatter(X[0:50,0], X[0:50,1],color ='g' )
    ax.scatter(X[50:,0], X[50:,1],color = 'b' )
    plt.title('Gaussian discriminant analysis')
    plt.xlabel('Ring Diameter in fresh water')
    plt.ylabel('Ring Diameter in marine water')
    plt.savefig('quad.png')	
    plt.show()


    

   
if (which_fun==0):
    partabc(X,Y)
else:
    partdef(X,Y)   
