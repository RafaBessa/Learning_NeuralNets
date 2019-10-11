import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import datetime
print(tf.__version__)


#Getting the data

Datadir = "/home/bessa/Downloads/kagglecatsanddogs_3367a/PetImages"

Categories = ["Dog","Cat"]
num_channels = 1
IMG_SIZE = 50
i =0
training_data = []
#transforming the data
for category in Categories:
    path = os.path.join(Datadir, category) # path
    class_img = Categories.index(category)
    
    for img in os.listdir(path):
        if i>5000: break;
        print(img)
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_img_array, class_img])
        except Exception as e:
            pass
        #plt.imshow(new_img_array,cmap = "gray")
        #plt.show()
        i+=1
        
    i=0
random.shuffle(training_data)
print(np.shape(training_data))

#training_data = np.array(training_data/255.0)
X = []
y = []

for p in training_data:
    X.append(p[0]/255.0) #NORMALIZE
    y.append(p[1])

    
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)



x_train = np.reshape(x_train, ((np.shape(x_train))[0],IMG_SIZE, IMG_SIZE,num_channels))
x_test = np.reshape(x_test, ((np.shape(x_test))[0],IMG_SIZE, IMG_SIZE,num_channels))

print(np.shape(x_train))
print(np.shape(y_test))
print(np.shape(Categories))

pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()


pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()