import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import random
from sklearn.model_selection import train_test_split
print(tf.__version__)

#Getting the data

Datadir = "/home/bessa/Downloads/kagglecatsanddogs_3367a/PetImages"

Categories = ["Dog","Cat"]
num_channels = 3
IMG_SIZE = 160
i =0
training_data = []
#transforming the data
for category in Categories:
    path = os.path.join(Datadir, category) # path
    class_img = Categories.index(category)
    
    for img in os.listdir(path):
        if i>100: 
            break
        #print(img)
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
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



IMG_SHAPE = (IMG_SIZE,IMG_SIZE, 3)

base_model = InceptionV3(input_shape = IMG_SHAPE, include_top = False, weights='imagenet')
base_model.trainable = False



#model = tf.keras.Sequential([base_model,keras.layers.GlobalAveragePooling2D(),Dense(1, activation='sigmoid')])
print(base_model.output)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(12, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#trainnig model

FitModel = model.fit(x_train,y_train,epochs=5,batch_size=256,validation_data=(x_test, y_test))



#validation

_, accuracy = model.evaluate(x_test,y_test)
print('Accuracy: %.2f' % (accuracy*100))



plt.figure(figsize=(20, 5))
plt.plot(FitModel.history['loss'], color='blue')
plt.plot(FitModel.history['val_loss'], color='red')
plt.title('Model loss', fontsize=20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Validação'], loc='upper right', fontsize=14)
plt.show()
