from __future__ import print_function
import time,sys,statistics,csv
import numpy as np 
import pandas as pd
from operator import itemgetter

sys.setrecursionlimit(30000)
val_accuracy=[]
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
validation=pd.read_csv('valid.csv')
def get_columns(df):
    return df.columns.values.tolist()
attributes=get_columns(train)

continuous=['X0','X1','X5','X12','X13','X14','X15','X16','X17',
                'X18','X19','X20','X21','X22','X23']
categorical=['X2','X3','X4','X6','X7','X8','X9','X10','X11','Y']

def get_unique(df,attribute):
    return list(df[attribute].unique())[1:]

train=train.drop(0)
validation=validation.drop(0)
test=test.drop(0)

train=train.astype('int64')
validation=validation.astype('int64')
test=test.astype('int64')

def encoding_categorical(df,categorical):
    for each in categorical:
        df[each]=df[each]-df[each].min()
    return df
train=encoding_categorical(train,categorical)
vaildation=encoding_categorical(validation,categorical)
test=encoding_categorical(test,categorical)

def encoding_continuous(df,continuous):
    for each in continuous:
        df[each] = np.where(df[each] > df.loc[:,each].median(), 1,0)    
    return df

train=encoding_continuous(train,continuous)
validation=encoding_continuous(validation,continuous)
test=encoding_continuous(test,continuous)
trainlabels=train.iloc[:,-1]
trainlabels=trainlabels.values
trainz=train.iloc[:,0:-1]
traindata=trainz.values

testlabels=test.iloc[:,-1]
testlabels=testlabels.values
testz=test.iloc[:,0:-1]
testdata=testz.values

validationlabels=validation.iloc[:,-1]
validationlabels=validationlabels.values
validationz=validation.iloc[:,0:-1]
validationdata=validationz.values

import collections
def entropy(labels):
    d=dict(collections.Counter(labels))
    n=len(indices)
    ent=0
    for v in d.values():
        probability=float(v)/n
        ent-=probability*math.log(probability)
        #ent=-ent
    return ent

def information_gain(labels, attr_list):    
    entr = entropy(labels)
    ig_max = 0
    d_max = None
    feature_index = None
    for i in range(24):   
        if attr_list[i] == 1:     
            feature = traindata[:,i]
            d = {}
            for ind in labels:
                if feature[ind] in d :
                    d[feature[ind]].append(ind)
                else :
                    d[feature[ind]] = [ind]
            if (d_max == None or feature_index == None):
                d_max = d
                feature_index = i
            net_ent = 0
            n=len(labels)
            for v in d.values():
                net_ent += ((float(len(v))/n) * entropy(v))
            information=entr - net_ent
            if (  information  > ig_max ):
                ig_max = information
                d_max = d
                feature_index = i
    return ig_max, feature_index, d_max

def get_accuracy(labels):
    pos = 0
    n=len(labels)
    for i in labels:
        if trainlabels[i] == 1:
            pos += 1
    percentage = 100.0 * float(pos) /n
    if (percentage < (100 - percentage)):
        return (0, 100.0-percentage)
    else:
        return (1, percentage)

import math, time, copy
import matplotlib.pyplot as plt

data_attributes = get_columns(train)
num_nodes,max_ht = 0,0
accuracy_train,accuracy_validation ,accuracy_test,last_list,count= [],[],[],[],[]


class Node:
    def __init__(self, child_d, child_if, f_split, height, indices, p, attr_index, parent):
        if (child_d == {}) :
            self.num_child = 0
        else:
            self.num_child = len(child_d.keys())
        
        self.indices = indices
        self.predicted = p
        self.f_split = f_split
        self.parent = parent
        self.child_if = child_if
        self.height = height     
        self.visited = 0  
        self.child_inds = child_d
        self.unused_attributes = attr_index
        self.child_nodes = {}
        

def tree( target_node ):
    global num_nodes, max_ht, last_list
    ig, feature_index, child_node_d = information_gain(target_node.indices, target_node.unused_attributes)
    if (ig == 0):
        target_node.child_if = 1
        last_list.append((target_node, target_node.height))
        return
    
    else:
        height_value=target_node.height + 1
        child_height = height_value
        for key, value in target_node.child_inds.items():     
            temp = copy.deepcopy(target_node.unused_attributes) 
            lasdf = []
            attr_index = temp
            temp = make_node(value, child_height, attr_index, target_node)
            lasdf = []
            cnode = temp
            temp = cnode
            lasdf = []
            target_node.child_nodes[key] = temp
            tree ( target_node.child_nodes[key] )
        return

def make_node(indices, height, attr_index, myparent):
    global num_nodes, max_ht
    global tree_root
    num_nodes += 1
    ig, feature_index, child_node_d = information_gain(indices, attr_index)
    acc = get_accuracy(indices)
    my_root = Node (child_node_d , 0, feature_index, height, indices, acc[0], attr_index, myparent)    
    if height > max_ht:
        max_ht = height
    return my_root

def one_data (target_node, data, label):
    if target_node.child_if == 1:
        if label != target_node.predicted:
            return 0
        else:
            return 1
    else:
        val = target_node.f_split
        my_val = data[val]
        if my_val in target_node.child_nodes:
            return one_data (target_node.child_nodes[my_val], data, label)
        else:
            if target_node.predicted != label:
                return 0
            else:
                return 1

def all_data (target_node, data, label):
    accuracy = 0
    for d, l in zip(data, label):
        accuracy =accuracy+ one_data (target_node, d, l)
    return (100 *float(accuracy) /len(label))

def pruning (p_list, prev_acc):
	print(len(p_list))
	global val_accuracy
	global num_nodes, accuracy_train, accuracy_validation, accuracy_test, count
	high_ht = max(p_list, key=itemgetter(1))[0]
	value = None
	x=high_ht.parent
	target_node = x
	for val, keys in target_node.child_nodes.items():
		if keys.indices == high_ht.indices:
			value = val
			break
	new_acc = all_data_data (tree_root, validationdata, validationlabels)
	high_ht = target_node.child_nodes[val]
	del target_node.child_nodes[val]
    
    
	if new_acc >= prev_acc:
		del target_node.child_inds[val]
		val_accuracy.append(new_acc)
		#print(len(p_list))
		if ( len(p_list)%5 == 0 ):
			accuracy_train.append(all_data_data (tree_root, traindata, trainlabels))
			accuracy_test.append(all_data_data (tree_root, testdata, testlabels))
			accuracy_validation.append(new_acc)
			count.append(len(p_list))
		x=target_node.num_child-1
		target_node.num_child =x
		fghfhj=[] 
		num_nodes =num_nodes- 1
		if len(target_node.child_nodes) != 0:
			ljhk=[]
		else:
			valu=1
			target_node.child_if = valu
			p_list.append((target_node, target_node.height))
		p_list.remove((high_ht, high_ht.height))
		if (1<len(p_list)):
			return pruning (p_list, new_acc)
		else:
			target_node.child_nodes[val] = high_ht
			p_list.remove((high_ht, high_ht.height))
		
		if len(p_list) >= 1:
			return pruning (p_list, prev_acc)
    

def main_func():
    print ("Number of nodes", num_nodes, max_ht )
    print ("Training accuracy", all_data (root_tree, traindata, trainlabels))
    print ("validation accuracy", all_data (root_tree, validationdata, validationlabels))
    print ("test accuracy", all_data (root_tree, testdata, testlabels))
    print(traindata.shape,testdata.shape,validationdata.shape)
    plt.plot(count, accuracy_train, color="green", label="Training")
    plt.plot(count, accuracy_validation, color="blue", label="Validation"),
    plt.plot(count, accuracy_test, color="red", label="Testing")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    indices = []
    for i in range(len(trainlabels)):
        indices.append(i)
    #print(indices)

    attribute_ind = []
    for i in range(24):
        attribute_ind.append(1)

    root_tree = make_node (indices, 0, attribute_ind, None)
    tree (root_tree)
    main_func()



