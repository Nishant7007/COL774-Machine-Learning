
import time,sys,statistics,csv
import numpy as np 
import pandas as pd
import collections
import math, time, copy
import matplotlib.pyplot as plt


continuous=['X0','X1','X5','X12','X13','X14','X15','X16','X17',
                'X18','X19','X20','X21','X22','X23']
categorical=['X2','X3','X4','X6','X7','X8','X9','X10','X11','Y']



train=pd.read_csv('credit-cards.train.csv')
test=pd.read_csv('credit-cards.test.csv')
validation=pd.read_csv('credit-cards.val.csv')
def data_columns(df):
    return df.columns.values.tolist()
attributes=data_columns(train)
def get_unique(df,attribute):
    return list(df[attribute].unique())[1:]



train=train.drop(0)
test=test.drop(0)
validation=validation.drop(0)
train=train.astype('int64')
test=test.astype('int64')
validation=validation.astype('int64')
def encoding_categorical(df,categorical):
    for each in categorical:
        df[each]=df[each]-df[each].min()
    return df
train=encoding_categorical(train,categorical)
test=encoding_categorical(test,categorical)
validation=encoding_categorical(validation,categorical)

def encoding_continuous(df,continuous):
    for each in continuous:
        median_val=df.loc[:,each].median()
        df[each] = np.where(df[each] > median_val, 1,0)    
    return df

train=encoding_continuous(train,continuous)
test=encoding_continuous(test,continuous)
validation=encoding_continuous(validation,continuous)

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

data_attributes = data_columns(train)
num_nodes = 0
max_ht = 0
accuracy_train = []
accuracy_validation = []
accuracy_test = []
count = []

def entropy(labels):
    d=dict(collections.Counter(labels))
    n=len(indices)
    ent=0
    for v in d.values():
        probability=float(v)/n
        ent-=probability*math.log(probability)
        #ent=-ent
    return ent
def information_gain(labels, attributes):    
    ent = entropy(labels)
    ig_max = 0
    d_max = None
    feature_index = None
    for i in range(24):   
        if attributes[i] == 1:     
            feature = traindata[:,i]
            d = {}
            
            for l in labels:
                if feature[l] in d :
                    d[feature[l]].append(l)
                else :
                    d[feature[l]] = [l]
            
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
            info=ent-net_ent
            if ( ( info ) > ig_max ):
                ig_max = info
                d_max = d
                feature_index = i
    return ig_max, feature_index, d_max

def get_accuracy(indices):
    pos = 0
    for i in indices:
        if trainlabels[i] == 1:
            pos += 1
    n=len(indices)
    percentage = 100.0 * float(pos) /n
    if ( percentage < (100 - percentage) ):
        return (0, 100.0-percentage)
    else:
        return (1, percentage)

class Node:
    def __init__(self, child_d, is_child, feature_to_split_on, height, indices, p, inds_attr, parent):
        if (child_d == {}) :
            self.num_child = 0
        else:
            self.num_child = len(child_d.keys())
        
        self.child_nodes = {}
        self.height = height
        self.is_child = is_child
        self.indices = indices
        self.feature_to_split_on = feature_to_split_on
        self.visited = 0 
        self.predicted = p 
        self.ununsed_attr = inds_attr
        self.parent = parent
        self.child_inds = child_d
        


# In[75]:


def grow_tree( target ):
    global num_nodes, max_ht
    
    ig, feature_index, child_node_d = information_gain(target.indices, target.ununsed_attr)
    if (ig == 0):
        target.is_child = 1
        return
    
    else:
        cheight = target.height + 1
        for key, value in target.child_inds.items():            
            inds_attr = copy.deepcopy(target.ununsed_attr)
            cnode = make_node(value, cheight, inds_attr, target)
            target.child_nodes[key] = cnode
            grow_tree ( target.child_nodes[key] )
        return


# In[76]:


def make_node(indices, height, inds_attr, myparent):
    global num_nodes, max_ht, accuracy_train, accuracy_validation, accuracy_test, count
    global root_tree
    num_nodes += 1
    ig, feature_index, child_node_d = information_gain(indices, inds_attr)
    acc = get_accuracy(indices)
    
    my_root = Node (child_node_d , 0, feature_index, height, indices, acc[0], inds_attr, myparent)    
    #if feature_index != None:
    #    my_root.ununsed_attr[feature_index] = 0
    
    if max_ht <height:
        max_ht = height
    
    if num_nodes > 1 and (num_nodes%30==0):
        #print (num_nodes)
        count.append(num_nodes)
        accuracy_train.append(full (root_tree, traindata, trainlabels))
        accuracy_validation.append(full (root_tree, validationdata, validationlabels))
        accuracy_test.append(full (root_tree, testdata, testlabels))
    return my_root


# In[77]:


def single (target, data, label):
    if target.is_child == 1:
        if label == target.predicted:
            return 1
        else:
            return 0
    else:
        val = target.feature_to_split_on
        my_val = data[val]
        if my_val in target.child_nodes:
            return single (target.child_nodes[my_val], data, label)
        else:
            if target.predicted == label:
                return 1
            else:
                return 0


# In[78]:


def full (target, data, label):
    acc = 0
    for d, l in zip(data, label):
        acc += single (target, d, l)
    return (100 * float(acc) / len(label))


# In[79]:
def main_func():
    print ("Number of nodes", num_nodes, max_ht )
    print ("Training accuracy", full (root_tree, traindata, trainlabels))
    print ("validation accuracy", full (root_tree, validationdata, validationlabels))
    print ("test accuracy", full (root_tree, testdata, testlabels))
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

    inds_attr = []
    for i in range(24):
        inds_attr.append(1)

    root_tree = make_node (indices, 0, inds_attr, None)
    grow_tree (root_tree)
    main_func()
    
    

