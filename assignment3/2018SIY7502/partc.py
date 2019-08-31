#!/usr/bin/env python
# coding: utf-8

# In[129]:


from __future__ import print_function
import time,sys,statistics,csv
import numpy as np 
import pandas as pd


# In[130]:


train=pd.read_csv('credit-cards.train.csv')
test=pd.read_csv('credit-cards.test.csv')
validation=pd.read_csv('credit-cards.val.csv')
def get_columns(df):
    return df.columns.values.tolist()
attributes=get_columns(train)


# In[131]:


continuous=['X0','X1','X5','X12','X13','X14','X15','X16','X17',
                'X18','X19','X20','X21','X22','X23']
categorical=['X2','X3','X4','X6','X7','X8','X9','X10','X11','Y']
categorical_list=[2,3,4,6,7,8,9,10,11,24]


# In[132]:


def get_unique(df,attribute):
    return list(df[attribute].unique())[1:]


# In[133]:


X2_l=get_unique(train,'X2')
X3_l=get_unique(train,'X3')
X4_l=get_unique(train,'X4')
X6_l=get_unique(train,'X6')
X7_l=get_unique(train,'X7')
X8_l=get_unique(train,'X8')
X10_l=get_unique(train,'X10')
X11_l=get_unique(train,'X11')
y_l=get_unique(train,'Y')


# In[134]:


train=train.drop(0)
test=test.drop(0)
validation=validation.drop(0)


# In[135]:


train=train.astype('int64')
test=test.astype('int64')
validation=validation.astype('int64')


# In[136]:


def encoding_categorical(df,categorical):
    for each in categorical:
        df[each]=df[each]-df[each].min()
    return df


# In[137]:


train=encoding_categorical(train,categorical)
test=encoding_categorical(test,categorical)
validation=encoding_categorical(validation,categorical)


# In[138]:


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


# In[139]:


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


# In[140]:


def information_gain(labels, attributes):    
    ent = entropy(labels)
    ig_max = 0
    d_max = None
    feature_index = None
    f_median=None
    max_median=None
    if sum(attributes)==1:
        return ig_max,feature_index,d_max,None
    for i in range(24):   
        if attributes[i] == 1:     
            feature = traindata[:,i]
            d = {}
            if i in categorical_list:
                for l in labels:
                    if feature[l] in d :
                        d[feature[l]].append(l)
                    else :
                        d[feature[l]] = [l]
            else:
                ind_features = []
                for ind in labels:
                    ind_features.append( traindata[ind][i] )
                
                for ind in labels:
                    temp = (float(feature[ind]) >= statistics.median(ind_features)) 
                    if temp in d :
                        d[temp].append(ind)
                    else :
                        d[temp] = [ind]
                
            
            if (d_max == None or feature_index == None):
                d_max = d
                feature_index = i
            
            net_ent = 0
            n=len(labels)
            for v in d.values():
                prob_x = float(len(v))/n
                ''' Calculating entropy over all the values of dictonary creates '''
                ent = entropy (v)
                net_ent += (prob_x * ent)
            info=ent - net_ent
            if ( ( info ) > ig_max ):
                ig_max = ( info )
                d_max = d
                feature_index = i
                max_median=f_median
    if feature_index not in categorical_list:
        return ig_max, feature_index, d_max, f_median
        
    else:
        return ig_max, feature_index, d_max,None


# In[141]:


def get_accuracy(indices):
    pos = 0
    for i in indices:
        if trainlabels[i] == 1:
            pos += 1
    
    pos_per = 100.0 * float(pos) / len(indices)
    if ( pos_per > (100 - pos_per) ):
        return (1, pos_per)
    else:
        return (0, 100.0-pos_per)


# In[142]:


import math, time, copy
import matplotlib.pyplot as plt

data_attributes = get_columns(train)
num_nodes = 0
max_ht = 0
train_acc = []
valid_acc = []
test_acc = []
last_list = []
count = []
scale=[]


# In[143]:


class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p, attribute_index, parent,med):
        if (child_d == {}) :
            self.num_child = 0
        else:
            self.num_child = len(child_d.keys())
        self.child_inds = child_d
        self.child_nodes = {}
        self.is_child = is_child
        self.split_feature = split_feature
        self.height = height
        self.indices = indices
        self.predicted = p    
        self.ununsed_attr = attribute_index
        self.parent = parent
        self.visited = 0 
        self.med_split_feature = med


# In[144]:


def grow_tree( target_node ):
    
    ig = information_gain(target_node.indices, target_node.ununsed_attr)[0]
    
    if (ig == 0):
        target_node.is_child = 1
        return
    
    else:
        cheight = target_node.height + 1
        for key, value in target_node.child_inds.items():            
            attribute_index = copy.deepcopy(target_node.ununsed_attr)
            cnode = make_node(value, cheight, attribute_index, target_node)
            target_node.child_nodes[key] = cnode
            grow_tree ( target_node.child_nodes[key] )
        return


# In[145]:


def make_node(indices, height, attribute_index, myparent):
    global num_nodes, max_ht, train_acc, valid_acc, test_acc, tree_root, scale
    num_nodes += 1
    ig, feature_index, child_node_d, feature_med = information_gain(indices, attribute_index)
    acc = get_accuracy(indices)
    my_root = Tree_Node (child_node_d , 0, feature_index, height, indices, acc[0], attribute_index, myparent, feature_med)
    if (feature_index != None and (feature_index in categorical_list)):
        my_root.ununsed_attr[feature_index] = 0
    
    if height > max_ht:
        max_ht = height
    
    if num_nodes > 1 and (num_nodes%20 == 0):
        #print (num_nodes)
        train_acc.append(all_data (tree_root, traindata, trainlabels))
        scale.append(num_nodes)
        valid_acc.append(all_data (tree_root, validationdata, validationlabels))
        test_acc.append(all_data (tree_root, testdata, testlabels))
    print (num_nodes, feature_index, feature_med)
    return my_root


# In[146]:


def one_data (target_node, data, label):
    if target_node.is_child == 1:
        return int(label == target_node.predicted)
    else:
        val = target_node.split_feature
        if val not in categorical_list:
            my_val = float(data[val]) >= target_node.med_split_feature
        else:
            my_val = data[val]
        
        if my_val in target_node.child_nodes:
            return one_data (target_node.child_nodes[my_val], data, label)
        else:
            return int(target_node.predicted == label)


# In[147]:


def all_data (target_node, data, label):
    acc = 0
    for d, l in zip(data, label):
        acc += one_data (target_node, d, l)
    return (100 * float(acc) / len(label))


# In[148]:



if __name__ == "__main__":
    indices = list(np.arange(0, len(trainlabels)))

    attribute_index = [1]*24

    tree_root = make_node (indices, 0, attribute_index, None)
    grow_tree (tree_root)
    plt.title("Plotting Accuracies vs Number of nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.plot(scale, train_acc, color="green", label="Training")
    plt.plot(scale, valid_acc, color="blue", label="Validation")
    plt.plot(scale, test_acc, color="red", label="Testing")
    plt.legend()
    plt.show()
    print(scale)
    print(test_acc)
    print(train_acc)
    print(valid_acc)

    print(len(scale),len(train_acc),len(test_acc),len(valid_acc))


# In[ ]:





# In[ ]:




