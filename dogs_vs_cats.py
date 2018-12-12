import numpy as np
from keras.optimizers import Adam
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

import cv2
import os
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

%matplotlib inline

# Data: https://www.kaggle.com/c/dogs-vs-cats/data
# View a cat and dog example
path = "D:/git/Dogs_vs_Cats/"
os.chdir(path)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
img_cat = cv2.imread('train2/cat.9.jpg')
plt.imshow(img_cat)
plt.axis('off')

fig.add_subplot(1, 2, 2)
img_dog = cv2.imread('train2/dog.0.jpg')
plt.imshow(img_dog)
plt.axis('off')

print(img_cat.shape)
print(type(img_cat))
print(img_dog.shape)

img_size = 60
# Resize input image to img_size of 60*60
def img_resize(img, img_size=img_size):
    old_size = img.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_img

new_img = img_resize(img_cat)
plt.imshow(new_img)

# prepare X and y
y = []
X = []
for f in glob.glob('train/*.jpg'):
    target = 1 if 'cat' in f else 0
    y.append(target)
    img = img_resize(cv2.imread(f), img_size=img_size)
    X.append(img)
X = np.array(X)
y = np.array(y).reshape(-1,1)
X= X/255  # Normalize input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print ("number of training examples: ", X_train.shape[0])
print ("number of test examples: ", X_test.shape[0])
print ("X_train shape: ", X_train.shape)
print ("Y_train shape: ", y_train.shape)
print ("X_val shape: ", X_val.shape)
print ("Y_val shape: ", y_val.shape)
print ("X_test shape: ", X_test.shape)
print ("Y_test shape: ", y_test.shape)

# Build CNN.
# Architecture source: https://www.kaggle.com/abhishekrock/cat-dog-try
def DogCatModel(input_shape):
    """  
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)

    X = Conv2D(kernel_size=(3,3),filters=3,activation="relu")(X_input)
    X = Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same")(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
    X = Conv2D(kernel_size=(3,3),filters=3,activation="relu")(X)
    X = Conv2D(kernel_size=(5,5),filters=5,activation="relu")(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Conv2D(kernel_size=(2,2),strides=(2,2),filters=10)(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    X = Dense(100, activation='sigmoid', name='fc1')(X)
    X = Dense(1, activation='sigmoid', name='fc2')(X)

    # Create model. This creates a Keras model instance, on which we will train/test the model.
    model = Model(inputs = X_input, outputs = X, name='DogCatModel')
    
    return model
	
dogcatModel = DogCatModel((img_size, img_size, 3))
# optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dogcatModel.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])
dogcatModel.fit(x = X_train, y = y_train, validation_data=(X_val, y_val), epochs = 20, batch_size = 10)

# Make prediction
preds = dogcatModel.evaluate(x = X_test, y = y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

dogcatModel.summary()