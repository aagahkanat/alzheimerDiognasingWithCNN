import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount("/content/gdrive")
data_src= "/content/gdrive/My Drive/ADNI_MRI_IMAGES/Dataset/"

data= []
y=[]
for d in os.listdir(data_src):
    for file in os.listdir(data_src+d):
        data.append(Image.open(data_src+d+'/'+file))
        y.append(d)
X=[]
for im in data:
    X.append(np.array(im))
X=np.array(X)
X.shape


X=X/255
non=0
mild=0
mod=0
vm=0
for i in y:
    if i=="Mild_Demented":
        mild+=1
    elif i=="Moderate_Demented":
        mod+=1
    elif i=="Non_Demented":
        non+=1
    else:
        vm+=1
print("Non Demented: ",non)
print("Very Mild: ",vm)
print("Moderate: ",mod)
print("Mild :",mild)
y_num=[]
for j in y:
    if j=="Mild_Demented":
        y_num.append(2)
    elif j=="Moderate_Demented":
        y_num.append(3)
    elif j=="Non_Demented":
        y_num.append(0)
    else:
        y_num.append(1)
    
y=to_categorical (y_num)
y.shape


X=X.reshape (6400, 128, 128,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=42)
print (X_train.shape, X_test.shape,y_train.shape,y_test.shape)


#Create the model
cnn = Sequential()
#initially 64 convolution nodes
cnn.add(Conv2D(64, (3,3), padding="same", activation='relu',input_shape=X_train.shape[1:]))
#add a Max Pooling layer
cnn.add(MaxPooling2D())
#another 32 convolution nodes
cnn.add(Conv2D(32, (3,3), padding="same", activation='relu'))
#Add a max pooling
cnn.add(MaxPooling2D())
#Add 32 convolutions
cnn.add(Conv2D(32, (2,2),padding="same", activation='relu'))
#Add a max pooling
cnn.add(MaxPooling2D())
#Flatten before adding fully connected layer
cnn.add(Flatten())
#Add a hidden layer with 100 nodes
cnn.add(Dense(100, activation='relu'))
#Add another hidden layer with 50 nodes
cnn.add(Dense(50, activation='relu'))
#Add final output layer with 4 output nodes using softmax
cnn.add(Dense(4, activation='softmax'))
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
cnn.summary()

history = cnn.fit(X_train, y_train, epochs =20, validation_data = (X_test, y_test))

y_pred =cnn.predict(X_test)
y_val = []

for y in y_pred:
  y_val.append(np.argmax(y))

y_true = []
for y in y_test:
  y_true.append(np.argmax(y))



print (classification_report(y_true,y_val))
print("Accuracy on test data: ",accuracy_score(y_true,y_val))
