import pandas as pd
import numpy as np
from sklearn 
from sklearn import svm
from sklearn.metrics import accuracy_score as ac
from keras.models import load_model
import numpy as np
import cv2 
from sklearn
import os as os
import cv2 
from sklearn





def create_data(directory):
    _, directories, _ = next(os.walk(directory))
    directories=sorted(directories)
    count = 0
    train_x, train_y,image_count_list = [],[],[]
    for folder in directories:
        if count==500:
            break
        count+=1
        print(count)
        subdirectory = directory+'/'+folder
        _, _, files = next(os.walk(subdirectory))
        image_count = 0
        files=sorted(files)
        
        for file in files:
            if('.png'==os.path.splitext(file)[1]):
                img = cv2.imread(subdirectory+'/'+file)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img=img.flatten()
                img=list(img)
                train_x.append(img)
                image_count += 1
            if('.csv'==os.path.splitext(file)[1]):
                df = pd.read_csv(subdirectory+'/'+file,header=None)
                df = np.array(df)
                df=df.flatten()
                df=df.astype(int)
                df=list(df)
                train_y.append(df)
        image_count_list.append(image_count)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x,train_y,image_count_list

train_dataset_directory = 'train_dataset'
train_x,train_y,list_image_count=create_data(train_dataset_directory)
print(train_x)
print(train_y)


pca=sklearn.decomposition.PCA(n_components=50)
pca.fit_transform(train_x)
train_x=pca.transform(train_x)

def final_data(image_count_list, train_x, train_y, limit=5006):
    final_train_x,final_train_y = [],[]
    for i in range(limit):
    	img_length = image_count_list[0]
        images_count = image_count_list[1:]
        temp_data_y = train_y[0]
        train_y = train_y[1:]
        temp_data_x = train_x[:img_length]
        train_x = train_x[img_length:]
        

        for j in range(img_length-7):
            temp = temp_data_x[j+6]
            temp1 = temp_data_x[j:j+6]
            groups = [[0,1],[0,2],[0,3],[0,4],[0,5],
            		[1,2],[1,3],[1,4],[1,5],[2,3],
            		[2,4],[2,5],[3,4],[3,5],[4,5]]
            
            for pair in groups:
                temp2 = [data for data in temp1]
                temp2.pop(pair[1])
                temp2.pop(pair[0])
                temp2.append(temp)
                temp2 = np.array(temp2)
                temp2=temp2.flatten()
                temp2=temp2.astype(float)
                temp2=list(temp2)
                final_train_x.append(temp2)
                final_train_y.append(temp_data_y[j+6])
    
    return np.array(final_train_x), np.array(final_train_y)

train_x, train_y = final_data(list_image_count, train_x, train_y)

model = svm.SVC()
model.fit(train_x, train_y)
predicted_y = model.predict(test_x)
accuracy = ac(train_y, predicted_y)
print('Guassian Kernel :', accuracy)
model = svm.svm.LinearSVC()
model.fit(train_x, train_y)
predicted_y = model.predict(test_x)
accuracy = ac(train_y, predicted_y)
print('Linear Kernel :', accuracy)

model.save('final_model')

#model = load_model('my_model.h5')



