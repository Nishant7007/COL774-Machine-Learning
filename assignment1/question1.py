import sys

input_file_path=sys.argv[1]
output_file_path=sys.argv[2]
learning_rate=sys.argv[3]
time_gap=sys.argv[4]

import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
from matplotlib import animation
X=np.genfromtxt(input_file_path).reshape((100,1))
Y=np.genfromtxt(output_file_path).reshape((100,1))
X=X-X.mean()-X.std()
Xorg=X
col=np.ones((100,1))
X=np.hstack((col,X))
theta=np.zeros((2,1))
htheta=np.matmul(X,theta)
def calc_jtheta(Y,htheta):
    error=np.square(np.subtract(Y,htheta))
    m=Y.shape[0]
    j_theta=(np.sum(error))/(2*m)
    return j_theta


def batch_gradiend(learning_rate,htheta,theta,X,Y):
    learning_rate=.001
    iter=1
    m=Y.shape[0]
    j_theta_list=[10**5,10**8]
    j_theta=0
    theta_zero=[]
    theta_one=[]
    while((iter <5000) and (abs(j_theta_list[iter]-j_theta_list[iter-1])>10**-12)):
        theta[0]=theta[0]-sum((htheta-Y))*learning_rate
        theta[1]=theta[1]-sum((htheta-Y)*Xorg)*learning_rate
        #print(theta[0])
        theta_zero.append(theta[0][0])
        theta_one.append(theta[1][0])
        iter+=1
        htheta=np.matmul(X,theta)
        j_theta=calc_jtheta(Y,htheta)
        j_theta_list.append(j_theta) 
    del j_theta_list[:2]
    #print(theta_one)
    return (j_theta, j_theta_list,theta_zero,theta_one)

def part_A():    
	j_theta, j_theta_list,theta_zero_list,theta_one_list=batch_gradiend(float(learning_rate),htheta,theta,X,Y)
	plt.plot(Xorg,Y,'.')
	plt.plot(Xorg, theta[0]+theta[1]*Xorg , '-')
	plt.show()   
	print(theta)


def cost_function(X,Y,theta):
    return float((1/100)*np.matmul(((Y-np.matmul(X,theta)).T),(Y-np.matmul(X,theta))))

x_range=np.arange(-1,3,0.05)
y_range=np.arange(-1,3,0.05)
z_mesh=np.zeros((len(x_range),len(y_range)))
lenx=len(x_range)
leny=len(y_range)

x_mesh,y_mesh=np.meshgrid(x_range,y_range)

for i in range(leny):
    for j in range(lenx):
        t=np.array([[x_mesh[i][j]],[y_mesh[i][j]]])
        z_mesh[i,j]=cost_function(X,Y,t)

j_theta, j_theta_list,theta_zero_list,theta_one_list=batch_gradiend(learning_rate,htheta,theta,X,Y)
def part_C(theta_zero_list,theta_one_list,j_theta_list):
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	surf=ax.plot_surface(x_mesh,y_mesh,z_mesh,linewidth=0.2,cmap='winter')
	ax.scatter(theta_zero_list,theta_one_list,j_theta_list,color='red')
	for i in range(len(theta_zero_list)):
	    plt.plot(theta_zero_list,theta_one_list,j_theta_list[i],'ro')
	    if(i%20==0):
	        plt.pause(0.2)
	    plt.title('sd')
	    plt.xlabel('theta_0')
	    plt.ylabel('theta_1')
	    ax.set_zlabel('cost function')
	plt.show() 

def part_D():
	CS=plt.contour(x_mesh,y_mesh,z_mesh,30)
	plt.colorbar(CS)
	plt.plot(theta_zero_list,theta_one_list,'-',color='red')
	for i in range(len(theta_zero_list)):
	    plt.plot(theta_zero_list[i],theta_one_list[i],'ro')
	    plt.title('fsdf')
	    plt.xlabel('Wine Acidity')
	    plt.xlabel('Wine Density')
	    plt.pause(float(time_gap))
	plt.title('plot for '+str(ll))
	plt.xlabel('Wine Density')
	plt.ylabel('Wine Density')
	plt.show()   


def part_E():
	ll=[0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]	     
	for l in ll:
		j_theta, j_theta_list,theta_zero_list,theta_one_list=batch_gradiend(l,htheta,theta,X,Y)
		part_D()
part_A()	
part_C(theta_zero_list,theta_one_list,j_theta_list)
part_D()
part_E()

