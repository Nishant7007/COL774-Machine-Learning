
import sys
train_file=str(sys.argv[1])
test_file=str(sys.argv[2])
part_num=str(sys.argv[3])


import json
import numpy
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot
import pandas as pd
import re
from collections import Counter
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
print('done')
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def split_it(str1):
    return re.sub(r'\W+', ' ', str1)


def flat_list(l):
    return [item for sublist in l for item in sublist]


def divider(df):
    for index, row in df.iterrows():
        i=int(row['stars'])
        l=(row['text'].split(' '))
        main_list[i-1].append(l) 
    for i in range(5):
        main_list[i]=flat_list(main_list[i])    
    return main_list

def divider_e(df):
    df=df.apply(lambda x: x.astype(str).str.lower())
    df['text']=df['text'].apply(split_it)
    main_list_e=[[],[],[],[],[]]
    for index, row in df.iterrows():
        if(index%10000==0):
            print(index)
        i=int(row['stars'])
        l=(row['text'].split(' '))
        main_list_e[i-1].append(l)
    for i in range(5):
        main_list_e[i]=flat_list(main_list_e[i]) 
    return main_list_e


def frequency_finder(l):
    freq_list=[]
    for i in range(5):
        counts=Counter(l[i])
        del counts['']
        freq_list.append(counts)
    return freq_list


def frequency_creator(freq_list):
    for i in range(len(freq_list)):
        with open('freq'+str(i+1)+'.pickle','wb') as f:
            pickle.dump(freq_list[i],f)
            
def frequency_creator_d(freq_list_d):
    for i in range(len(freq_list_d)):
        with open('freq_d_'+str(i+1)+'.pickle','wb') as f:
            pickle.dump(freq_list_d[i],f)            

def probability_creator():
    for i in range(5):
        pickle_in=open('freq'+str(i+1)+'.pickle','rb')
        ex=pickle.load(pickle_in)
        total=sum(ex.values(),0.0)
        for key in ex:
            ex[key]/=total
        print(sum(ex.values()))    
        with open('prob'+str(i+1)+'.pickle','wb') as f:
            pickle.dump(ex,f) 
            
def prior_prob(df):
    k=df['stars'].value_counts().to_dict()
    total=sum(k.values(),0.0)
    for key in k:
        k[key]/=total
    return k 

def word_class_frequency_finder(main_list):
    word_class_frequency=np.zeros(5)
    for i in range(len(main_list)):
        word_class_frequency[i]=len(main_list[i])
    return word_class_frequency    

def posterior(dk,words_prob_list,prior):
    dk['predicted']=0
    for index, row in dk.iterrows():
        print(index)
        words=(row['text'].split(' ')) 
        final=np.zeros(5)
        for c in range(len(words_prob_list)):
            prob_val=0.0
            for word in words:
                prob_val+=np.log(words_prob_list[c]['word'])
            final[c]=prob_val
            #+np.log(prior[c]) 
        print(final)
        #max_index = np.argmax(final)
        #print(max_index+1)
        #row['predicted']=max_index+1
    return dk

def test_data_a(test,words_freq,prior):    
    test=test.apply(lambda x: x.astype(str).str.lower())
    test['text']=test['text'].apply(split_it)
    test['predict']=0
    leng=len(words_freq)
    class_freq=np.zeros(5)
    for k,v in words_freq.items():
        for i in range(5):
            class_freq[i]+=words_freq[k][i]
    for index,row in test.iterrows():
        #print(index)
        words=(row['text'].split(' '))
        #final=np.zeros(5)
        #counts = Counter(words)
        final=np.zeros(5)
        for c in range(5):
            prob_val=0
            for word in words:
                if words_freq[word]==0:
                    x=1/(class_freq[c]+leng)
                else:    
                    x=(words_freq[word][c]+1)/(class_freq[c]+leng)
                prob_val+=np.log(x)
            prob_val+=np.log(prior[c])    
            final[c]=prob_val
        test.iat[index,2]=np.argmax(final)+1
    c=0
    for i in range(test.shape[0]):
        if (test.iloc[i]['stars']==str(test.iloc[i]['predict'])):
            c+=1
    print('test accuracy: '+str(c/test.shape[0]*100)) 
    return test.loc[: , "predict"]

def part_C(test,train):
	train= train[['text','stars']]
	test= test[['text','stars']]
	k=train['stars'].value_counts().to_dict()
	prior=[]
	for i,v in k.items():
		prior.append(v)
	pickle_in=open('freq.pickle','rb')
	words_freq=Counter(pickle.load(pickle_in))
	predicted_a=test_data_a(test,words_freq,prior) 	
	predicted_list = predicted_a.values.tolist()
	true_list = test['stars'].values.tolist()
	plot_confusion_matrix(true_list, predicted_list, classes= [1,2,3,4,5], title='Confusion matrix, without normalization')




def train_data_a(train,words_freq,prior):
    train=train.apply(lambda x: x.astype(str).str.lower())
    train['text']=train['text'].apply(split_it)
    train['predict']=0
    for index,row in train.iterrows():
        #print(index)
        words=(row['text'].split(' '))
        #final=np.zeros(5)
        final=np.zeros(5)
        for c in range(5):
            prob_val=0
            for word in words:
                if words_freq[word]==0:
                    x=1/(class_freq[c]+leng)
                else:    
                    x=(words_freq[word][c]+1)/(class_freq[c]+leng)
                prob_val+=np.log(x)
            prob_val+=np.log(prior[c])    
            final[c]=prob_val
        train.iat[index,2]=np.argmax(final)+1
    c=0
    for i in range(train.shape[0]):
        if (train.iloc[i]['stars']==str(train.iloc[i]['predict'])):
            c+=1
            #print(c)
    print('train accuracy: '+str(c/train.shape[0]*100))
    return train.loc[: , "predict"]

def part_A(test,train):
    #data =pd.read_json('train.json',lines=True) 
    train= train[['text','stars']]
    #test =pd.read_json('test.json',lines=True) 
    test= test[['text','stars']]
    k=train['stars'].value_counts().to_dict()
    prior=[]
    for i,v in k.items():
        prior.append(v)
    pickle_in=open('freq.pickle','rb')
    words_freq=Counter(pickle.load(pickle_in))    
    predicted_a=test_data_a(test,words_freq,prior)    
    predictions=predicted_a.tolist()
    labels=test['stars'].tolist()
    cm = confusion_matrix(labels, predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print('recall='+str(np.mean(recall)))
    print('precision='+str(np.mean(precision)))
    prec=np.mean(precision)
    rec=np.mean(recall)
    f1= 2 * (precision * recall) / (precision + recall)
    print("f1 score: "+str(f1))


def test_data_b(test,prior):
    test['predict']=0
    print(test)
    #print(type(test.iloc[0]['predict']))
    #print(type(test.iloc[0]['stars']))
    #print(type(test['predict']))
    #print(type(test['stars']))
    class_list=[1,2,3,4,5]
    for index,row in test.iterrows():
        row['predict']=random.choice(class_list)
        #print(row)    
    c=0
    for i in range(test.shape[0]):
        if (str(test.iloc[i]['stars'])==str(test.iloc[i]['predict'])):
            c+=1
            print(c)
    print('test accuracy using random class: '+str(c/test.shape[0]*100)) 
    labels=test['stars'].tolist()
    predictions=test['predict'].tolist()
    cm = confusion_matrix(labels, predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print('macro f1-score using random class: recall='+str(np.mean(recall)))
    print('macro f1-score using random class: precision='+str(np.mean(precision)))
    #test.drop(['predict'])
    test['predict1']=0
    top_class=np.argmax(prior_prob)+1
    for index,row in test.iterrows():
        row['predict1']=top_class
    c=0
    for i in range(test.shape[0]):
        if (test.iloc[i]['stars']==test.iloc[i]['predict1']):
            c+=1
    print('test accuracy using majority class: '+str(c/test.shape[0]*100)) 
    labels=test['stars'].tolist()
    predictions=test['predict1'].tolist()
    cm = confusion_matrix(labels, predictions)
    cm+=1
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print('macro f1-score using majority class: recall='+str(np.mean(recall)))
    print('macro f1-score using majority class: precision='+str(np.mean(precision)))
    #return test.loc[: , "predict"]
def part_B(test,train):
    k=train['stars'].value_counts().to_dict()
    prior=[]
    for i,v in k.items():
        prior.append(v)
    print(prior)    
    pred=test_data_b(test,prior)    
    
def test_data_d(test,words_freq,prior):
    lemmatizer = WordNetLemmatizer() 
    test['predict']='0'
    test=test.apply(lambda x: x.astype(str).str.lower())
    test['text']=test['text'].apply(split_it)
    stop_words = set(stopwords.words('english'))
    leng=len(words_freq)
    class_freq=np.zeros(5)
    for k,v in words_freq.items():
        for i in range(5):
            class_freq[i]+=words_freq[k][i]
    for index,row in test.iterrows():
        #print(index)
        words=(row['text'].split(' '))
        filtered_words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words]
        #final=np.zeros(5)
        #counts = Counter(words)
        final=np.zeros(5)
        for c in range(5):
            prob_val=0
            for word in filtered_words:
                if words_freq[word]==0:
                    x=1/(class_freq[c]+leng)
                else:    
                    x=(words_freq[word][c]+1)/(class_freq[c]+leng)
                prob_val+=np.log(x)
            prob_val+=np.log(prior[c])    
            final[c]=prob_val
        test.iat[index,2]=np.argmax(final)+1        
    c=0
    for i in range(test.shape[0]):
        if (test.iloc[i]['stars']==str(test.iloc[i]['predict'])):
            c+=1
    print('test accuracy: '+str(c/test.shape[0]*100)) 
    return test.loc[: , "predict"] 

def part_D(test,train):
    #data =pd.read_json('train.json',lines=True) 
    train= train[['text','stars']]
    #test =pd.read_json('test.json',lines=True) 
    test= test[['text','stars']]
    k=train['stars'].value_counts().to_dict()
    prior=[]
    for i,v in k.items():
        prior.append(v)
    pickle_in=open('freq_d.pickle','rb')
    words_freq=Counter(pickle.load(pickle_in))    
    predicted_a=test_data_d(test,words_freq,prior)    
    predictions=predicted_a.tolist()
    labels=test['stars'].tolist()
    cm = confusion_matrix(labels, predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print('recall='+str(np.mean(recall)))
    print('precision='+str(np.mean(precision)))
    #print(precision='+str(np.mean(precision)))

    prec=np.mean(prec)
    rec=np.mean(recall)
    f1= 2 * (precision * recall) / (precision + recall)
    print("f1 score: "+str(f1))

def test_data_e_1(test,words_freq,prior):
    lemmatizer = WordNetLemmatizer() 
    test['predict']='0'
    test=test.apply(lambda x: x.astype(str).str.lower())
    test['text']=test['text'].apply(split_it)
    stop_words = set(stopwords.words('english'))
    leng=len(words_freq)
    class_freq=np.zeros(5)
    for k,v in words_freq.items():
        for i in range(5):
            class_freq[i]+=words_freq[k][i]
    for index,row in test.iterrows():
        #print(index)
        words=(row['text'].split(' '))
        filtered_words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words]
        #final=np.zeros(5)
        #counts = Counter(words)
        final=np.zeros(5)
        bigram_words=[]
        for i in range(0,len(filtered_words) -1):
            bigram  = " ".join(words[i:i+2] )
            bigram_words.append(bigram)
        for c in range(len(class_freq)):
            prob_val=0
            for word in bigram_words:
                if words_freq[word]==0:
                    x=1/(class_freq[c]+leng)
                else:    
                    x=(words_freq[word][c]+1)/(class_freq[c]+leng)
                prob_val+=np.log(x)
                #print(word)
                #if word not in words_prob_list_e[c].keys():
                 #   x=1/5
                #else:    
                 #   x=(words_prob_list_e[c][word]+1)/(class_frequency[c]+5)
                #print(x)
                #prob_val+=np.log(x)
            prob_val+=np.log(prior[c])    
            final[c]=prob_val
        test.iat[index,2]=np.argmax(final)+1
    print('done')    
    c=0
    for i in range(test.shape[0]):
        if (test.iloc[i]['stars']==str(test.iloc[i]['predict'])):
            c+=1
            #print(c)
    print('test accuracy: '+str(c/test.shape[0]*100))    
    return test.loc[: , "predict"]




def test_data_e_2(test,words_freq,prior):
    lemmatizer = WordNetLemmatizer() 
    test['predict']='0'
    test=test.apply(lambda x: x.astype(str).str.lower())
    test['text']=test['text'].apply(split_it)
    stop_words = set(stopwords.words('english'))
    leng=len(words_freq)
    class_freq=np.zeros(5)
    for k,v in words_freq.items():
        for i in range(5):
            class_freq[i]+=words_freq[k][i]
    for index,row in test.iterrows():
        #print(index)
        words=(row['text'].split(' '))
        #filtered_words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words]
        #final=np.zeros(5)
        #counts = Counter(words)
        final=np.zeros(5)
        bigram_words=[]
        for i in range(0,len(words) -1):
            bigram  = " ".join(words[i:i+2] )
            bigram_words.append(bigram)
        for c in range(len(class_freq)):
            prob_val=0
            for word in bigram_words:
                if words_freq[word]==0:
                    x=1/(class_freq[c]+leng)
                else:    
                    x=(words_freq[word][c]+1)/(class_freq[c]+leng)
                prob_val+=np.log(x)
                #print(word)
                #if word not in words_prob_list_e[c].keys():
                 #   x=1/5
                #else:    
                 #   x=(words_prob_list_e[c][word]+1)/(class_frequency[c]+5)
                #print(x)
                #prob_val+=np.log(x)
            prob_val+=np.log(prior[c])    
            final[c]=prob_val
        test.iat[index,2]=np.argmax(final)+1
    print('done')    
    c=0
    for i in range(test.shape[0]):
        if (test.iloc[i]['stars']==str(test.iloc[i]['predict'])):
            c+=1
            #print(c)
    print('test accuracy: '+str(c/test.shape[0]*100))    
    return test.loc[: , "predict"]

def part_E(test,train):
    #data =pd.read_json('train.json',lines=True) 
    train= train[['text','stars']]
    #test =pd.read_json('test.json',lines=True) 
    test= test[['text','stars']]
    k=train['stars'].value_counts().to_dict()
    prior=[]
    for i,v in k.items():
        prior.append(v)
    pickle_in=open('freq_e_bigram.pickle','rb')
    words_freq=Counter(pickle.load(pickle_in))  
    print('lemmatization,stopwords and bigram')  
    predicted_a=test_data_e_1(test,words_freq,prior)    
    predictions=predicted_a.tolist()
    labels=test['stars'].tolist()
    cm = confusion_matrix(labels, predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print('recall='+str(np.mean(recall)))
    print('precision='+str(np.mean(precision)))
    prec=np.mean(precision)
    rec=np.mean(recall)
    f1= 2 * (prec * rec) / (prec + rec)
    print("f1 score: "+str(f1))

    # print('stopwords and bigram')
    # predicted_a=test_data_e_2(test,words_freq,prior)    
    # predictions=predicted_a.tolist()
    # labels=test['stars'].tolist()
    # cm = confusion_matrix(labels, predictions)
    # recall = np.diag(cm) / np.sum(cm, axis = 1)
    # precision = np.diag(cm) / np.sum(cm, axis = 0)
    # print('recall='+str(np.mean(recall)))
    # print('precision='+str(np.mean(precision)))
    # prec=np.mean(precision)
    # rec=np.mean(recall)
    # f1= 2 * (prec * rec) / (prec + rec)
    # print("f1 score: "+str(f1))







train =pd.read_json(train_file,lines=True) 
train= train[['text','stars']]
test =pd.read_json(test_file,lines=True) 
test= test[['text','stars']]

if part_num=="a":
	part_A(test,train)

if part_num=="b":
	part_B(test,train)	

if part_num=="c":
	part_C(test,train)

if part_num=="d":
	part_D(test,train)

if part_num=="e":
	part_E(test,train)	

