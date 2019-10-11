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
import pickle


Datadir = "/home/bessa/Downloads/kagglecatsanddogs_3367a/PetImages"

Categories = ["Dog","Cat"]
num_channels = 1
IMG_SIZE = 50
num_class = 2
#lodData
x_test = np.array(pickle.load(open("x_test.pickle","rb")))
print(np.shape(x_test))
x_train = np.array(pickle.load(open("x_train.pickle","rb")))
print(np.shape(x_train))

y_test = np.array(pickle.load(open("y_test.pickle","rb")))
print(np.shape(y_test))

y_train = np.array(pickle.load(open("y_train.pickle","rb")))
print(np.shape(y_train))


#config rede
Convn_layers = [1,2,3]
Convn_size = [32, 48, 64]
Dense_layers = [0,1,2]
Dense_size = [32,48,64] 

for conv in Convn_layers:
    for conv_size in Convn_size:
        for dense in Dense_layers :
            for dense_size in Dense_size:
                #Setting TensoBoard
                Name = "CatsAndDogs"+"-cnn-"+str(conv_size)+"x"+str(conv)+"-dense-"+str(dense_size)+"x"+str(dense) + str(datetime.datetime.now())
                print(Name)
                #logdir = "logs/"+Name
                logdir = os.path.join("logs",Name)
                tensorboard = TensorBoard(log_dir=logdir)
                #creating the model
                model = Sequential()
                
                for i in range(conv-1):
                    model.add(Conv2D(filters = conv_size, kernel_size=(3,3), activation = 'relu', input_shape=(IMG_SIZE, IMG_SIZE ,1)))
                    model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Flatten())
                
                for i in range(dense):
                    model.add(Dense(dense_size,activation='relu'))

                
                model.add(Dense(1, activation='sigmoid'))


                #compile
                model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

                #trainnig model

                FitModel = model.fit(x_train,y_train,epochs=5,batch_size=256,validation_data=(x_test, y_test,), callbacks= [tensorboard] )


#validation
