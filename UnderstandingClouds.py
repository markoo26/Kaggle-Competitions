### Import the libraries ###

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D, 
                         Flatten, Dense)

import os, shutil, sys, webbrowser

#for p in sys.path:
#    print(p)

os.chdir("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Clouds\\understanding_cloud_organization\IMG")
#webbrowser.open_new_tab("https://www.kaggle.com/c/understanding_cloud_organization")

train_labels = pd.read_csv("train.csv")
train_labels.head(30)

### Initialize CNN

classifier = Sequential()

### Step 1 - Convolution ###

classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), 
                             activation = 'relu'))

### Step 2 - Pooling ###

classifier.add(MaxPooling2D(pool_size = (2,2)))

### Step 3 - Flattening ###

classifier.add(Flatten())

### Step 4 - Full Connection ###

### how many nodes are in the hidden layer?
classifier.add(Dense(output_dim = 128, activation = 'relu' )) 
### probabilities of classes
classifier.add(Dense(output_dim = 4, activation = 'sigmoid' )) 

### Compiling the CNN ###

### instead of binary, because we have more categorical outcomes
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

### Image Preprocessing
### To prevent overfitting - we need to develop our image dataset so 
### that we have good accuracies on both training and test set


### Directory Preparation 

### pousuwac te, ktore w Encoded Pixels maja nan
### uruchomic petle, ktora przekopiuje wszystkie obrazki do wlasciwego folderu
### wywalic stare

train_labels["cloud_label"] = train_labels["Image_Label"].str[12:] ### WAZNE ### !
train_labels["file_name"] = train_labels["Image_Label"].str[:11]
train_labels = train_labels.dropna()

os.getcwd()

for index, row in train_labels.iterrows():
    src = "train_images/" + row["file_name"]
    dest = "train_images/" + row["cloud_label"]  + "/" + row["file_name"]
    shutil.copyfile(src,dest)

train_directory = "C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Clouds\\understanding_cloud_organization\\IMG\\train_images"

for path in os.listdir(train_directory):
    full_path = os.path.join(train_directory, path)
    if os.path.isfile(full_path):
        os.remove(full_path)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'train_images/',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory(
                                                'test_images/',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='categorical')


classifier.fit_generator(
        training_set,
        steps_per_epoch=500, ### number of images in the training set
        epochs=5,
        validation_data=test_set,
        validation_steps=2000) 