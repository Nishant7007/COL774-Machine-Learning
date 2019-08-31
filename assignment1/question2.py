#QUESTION-2
#IMPORTING ALL LIBRARIES

import sys

input_file_path=sys.argv[1]
output_file_path=sys.argv[2]
tau=sys.argv[3]


import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
from matplotlib import animation
from numpy.linalg import pinv
import math

#LOADING DATA AND PREPROCESSING

X=np.genfromtxt(input_file_path).reshape((100,1))
Y=np.genfromtxt(output_file_path).reshape((100,))
X=(X-X.mean())/X.std()
Xorg=X.reshape((100,))
col=np.ones((100,1))
X=np.hstack((col,X))
theta=np.zeros((2,1))


#FUNCTION FOR LINEAR REGRESSION
def linear_regression(X,Y):
        theta=pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(Y))
        return theta
points=np.ogrid[np.min(X[:,1]):np.max(X[:,1]):100j] 

#WEIGHT FUNCTION FOR LOCALLY WEIGHTED LINEAR REGRESSION
def weight(x,t):
    return np.diag(np.exp((-1) * ((x-X[:,1])**2) / (2 * t**2)))

#FUNCTION FOR LOCALLY WEIGHTED LINEAR REGRESSION

def local_regression(t):    
    result=[]
    for i in points:
        diag_W = weight(i,t)
        theta = np.matmul(pinv(np.transpose(X).dot(diag_W).dot(X)),np.transpose(X).dot(diag_W).dot(Y)) 
        result.append(np.matmul(theta,np.array([1, i])))
        plt.scatter(X[:,1],Y)
    result=np.array(result)
    plt.plot(points,result,'-r',)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title(' locally weighted linear regression with tau value= '+str(t))
    plt.gca().legend(('Data points','hypothesis function'))
    plt.savefig('tau='+str(t)+'.jpeg')
    return theta

#PART A FUNCTION    
def part_A():
    theta=linear_regression(X,Y)
    print("theta values are:"+str(theta))
    #plt.plot(X[:,1],Y,'.')
    #plt.plot(X[:,1], j_theta , '-')
    #plt.xlabel('x values')
    #plt.ylabel('y values')
    #plt.title('linear regression')
    #plt.gca().legend(('data points','hypothesis function'))
    #plt.savefig('q2pAqqq.jpg')
    #plt.show()
     

#PART B FUNCTION    
def part_B(t):
    local_regression(t) 
    plt.show()
    
#PART C FUNCTION    
def part_c():
    l=[0.1,0.3,2,10,0.05]
    for i in l:
        part_B(i)
        plt.show()   

#CALLING ALL FUNCTIONS

part_A()
part_B(float(tau))
#part_c()