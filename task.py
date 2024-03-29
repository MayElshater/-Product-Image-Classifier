# -*- coding: utf-8 -*-
"""Task.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iCARMw7GnPGvcaY_3VMMngaGCcFjcAPP

***This Project for Product Image Classifier***

# *Import*
"""

import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
import matplotlib.pyplot as plt
from keras.utils import plot_model
from zipfile import ZipFile
from google.colab import drive
import glob as gb

"""# Read data"""

drive.mount('/content/drive')

with ZipFile ('drive/MyDrive/task.zip','r') as ZipObj:
  ZipObj.extractall('drive/MyDrive/task')

# This line sets the path to the training data directory
train_path='drive/MyDrive/task/task/train/'

# This line sets the path to the testing data directory
test_path='drive/MyDrive/task/task/test/'

# Loop through each subdirectory in the train_path directory
for folder in os.listdir(train_path):

    # Use glob to find all the .jpg files in the current subdirectory
    images = gb.glob(pathname=str(train_path + folder + '/*.jpg'))

    # Print out the number of images found in the current subdirectory
    print(f'for training data, found {len(images)} in folder {folder}')

# Loop through each subdirectory in the test_path directory
for folder in os.listdir(test_path ):

    images=gb.glob(pathname=str(test_path  + folder +'/*.jpg'))

    print(f'for test data, found {len(images)} in folder {folder}')

# This code reads in all JPEG images in each folder of the train_path directory and finds the unique shapes of the images and their corresponding counts.
#It then prints out each unique shape and its count.
#and that help us know what the inputshape that we will give it to the model.
#if we have a different size of image or not becouse we need to make all same size.
#This for train dataset.
size=[]
for folder in os.listdir(train_path):
    images=gb.glob(pathname=str(train_path + folder +'/*.jpg'))
    for img in images:
        img=plt.imread(img)
        size.append(img.shape)
pd.Series(size).value_counts()

#This for test dataset.
size=[]
for folder in os.listdir(test_path):
    images=gb.glob(pathname=str(test_path + folder +'/*.jpg'))
    for img in images:
        img=plt.imread(img)
        size.append(img.shape)
pd.Series(size).value_counts()

code={'Accessories':0 ,'Artifacts':1,'Beauty':2,'Fashion':3,'Games':4,'Home':5,'Nutrition':6,'Stationary':7}
def getname(n):

    for k,v in code.items():
     if v==n:
        return k

#read train data, and split train into (x,y)
x_train=[]
y_train=[]
for folder in os.listdir(train_path ):
    images=gb.glob(pathname=str(train_path  + folder +'/*.jpg'))
    for img in images:
        img=cv.imread(img)
        x_train.append(img)
        y_train.append(code[folder])

#read test data, and split test data into (x.y)
x_test=[]
y_test=[]
for folder in os.listdir(test_path ):
    images=gb.glob(pathname=str(test_path  + folder +'/*.jpg'))
    for img in images:
        img=cv.imread(img)
        x_test.append(img)
        y_test.append(code[folder])

#visualization of a random selection of images from the x_train dataset, with their corresponding labels.
plt.figure(figsize=(20,20))
for i,v in enumerate(np.random.randint(0,len(x_train),36)):
    plt.subplot(6,6,i+1)
    plt.imshow(x_train[v])
    plt.title(getname(y_train[v]))
    plt.axis('off')

"""#Split Dataset"""

height,width=1600, 720
input_shape=(1600, 720, 3)
num_classes = 8
batch_size=4

# Load the training set from the specified directory, with a 90/10 train/validation split
train_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,   # 10% of the data will be used for validation
    subset="training",      # Use the training subset of the data
    seed=123,               # Set a random seed for reproducibility
    image_size=(height,width),  # Resize the images to the specified height and width
    batch_size=batch_size   # Set the batch size for training
)

# Load the validation set from the same directory, with the same 90/10 split
validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,   # 10% of the data will be used for validation
    subset="validation",    # Use the validation subset of the data
    seed=123,               # Set a random seed for reproducibility
    image_size=(height, width), # Resize the images to the specified height and width
    batch_size=batch_size   # Set the batch size for validation
)

# Load the test set from the specified directory
test_set = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(height, width), # Resize the images to the specified height and width
    batch_size=batch_size   # Set the batch size for testing
)

"""# EarlyStop And Checkpoint"""

# This for sets up early stopping during training.
#'val_loss' is monitored, and training stops if it doesn't improve for 8 epochs.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)

#This for sets up model checkpointing during training.
#'val_accuracy' is monitored, and the best model is saved to 'best_model.h5'.
#Only the best model is saved, based on validation accuracy.
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

"""# Model Building"""

# Import the EfficientNetV2B3 model from Keras applications
imported_model = tf.keras.applications.EfficientNetV2B3(include_top=False,
                                                        input_shape=input_shape,
                                                        pooling='max',
                                                        classes=8,
                                                        weights='imagenet')

# Create a new sequential model
model = Sequential()

# Add the imported model as a layer to the new model
model.add(imported_model)

# Add a flatten layer to the new model
model.add(Flatten())

# Add a dense layer with softmax activation and L2 regularization to the new model
model.add(Dense(8, activation='softmax', kernel_regularizer=regularizers.l2(0.1)))

# Compile the model with the following configurations:
model.compile(
    # Use the Adam optimizer with a learning rate of 0.001
    optimizer=Adam(lr=0.001),
    # Use sparse categorical crossentropy as the loss function
    loss='sparse_categorical_crossentropy',
    # Track accuracy as a metric during training and evaluation
    metrics=['accuracy']
)

#summary of the model
model.summary()

"""# Training $ Validation"""

# Train the EfficientNet model for 15 epochs using the training set and validation set,
# and use the EarlyStopping and ModelCheckpoint callbacks to stop training early
# and save the best model based on validation accuracy
history = model.fit(x = train_set,epochs = 15,validation_data = validation_set,callbacks=[es, mc])

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2)

# Get the training accuracy and loss from the history object
train_acc = history.history['accuracy']
train_loss = history.history['loss']

# Set the size of the figure
fig.set_size_inches(12, 4)

# Plot the training accuracy and validation accuracy on the first subplot
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

# Plot the training loss and validation loss on the second subplot
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

# Show the plot
plt.show()

# Evaluate the model on the training set and get the loss and accuracy
train_loss, train_acc = model.evaluate(train_set)

# Evaluate the model on the validation set and get the loss and accuracy
test_loss, test_acc = model.evaluate(validation_set)

# Print the final training accuracy and validation accuracy
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))

# Load the saved model from the file 'best_model.h5'
model = load_model('best_model.h5')

"""# Testing"""

# Use the loaded model to make predictions on the test set
y_pred = model.predict(test_set)

# Convert the predicted probabilities to class labels
y_pred = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the model on the test set and get the loss and accuracy
test_loss, test_acc = model.evaluate(test_set)

# Print the final test accuracy
print("Final test accuracy:", test_acc)

# Convert class labels from one-hot encoding to integers for the test set
y_test_int = np.concatenate([y.numpy() for x, y in test_set], axis=0)

# Print classification report
print(classification_report(y_test_int, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_int, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

