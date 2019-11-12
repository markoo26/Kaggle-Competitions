### To do:
# 1. What is padding?
# 2. Check other activation functions (once relu once softmax)
# 3. What is Strides parameter in MaxPooling?
# 4. What is Dropout (once 0.5 and other time 0.25)
# 5. Why in the beginning section with 5x5 kernel and then 3x3?
# 6. RMSProp optimizer?
# 7. Learning Rate reduction

#### Import of the libraries ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, webbrowser, time, random

#### Modelling libraries ####

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#### Import the datasets and preparation####

training_set = pd.read_csv("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Kannada-MNIST\\train.csv")
test_set = pd.read_csv("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Kannada-MNIST\\test.csv")

y_train = training_set["label"]
training_set = training_set.drop(['label'], axis=1)

#training_set["number_id"] = range(0,len(training_set))

#### Check_pictures object definition ####
#### Call the object to see either first n or random n pictures from the dataset ####
#### With observation number and predicted label ####

### Below: show show first (not randomly chosen) 15 records labelled as "3" from the training _set

check_pictures(dataset= training_set, labels = y_train,
               no_of_pictures=15, randomize = False, y_label = 5)

#    dataset = training_set ###
#    labels = y_train ### where the labels are stored
#    no_of_pictures = 15 ### how many pictures should be shown
#    randomize = False ### determines if the random/first {no_of_pictures} pictures are shown
#    y_label = 5 ### pictures for which number are going to be checked
    
def check_pictures(dataset, labels, no_of_pictures, randomize = False, 
                   y_label = None):

    fig = plt.figure()
    sp_index = 1

    if y_label:
        image_set = dataset[labels == y_label] ### take records only for determined number
    else:
        image_set = dataset### take records for all numbers
        
    if randomize:
        pictures_ids = np.random.randint(1, len(image_set), size=(1, no_of_pictures)) ### generate random picture ID's 
    else:
        pictures_ids = np.array(range(0,no_of_pictures)) ### create range from 0 to {no_of_pictures}

    for i in np.nditer(pictures_ids): ### build the plot with summary        
        
        indexer = int(i) ### extract the ID
        new_set = image_set.iloc[indexer,:].values.reshape(28,28) ### reshape the values from i-th row
        ax = fig.add_subplot(3,5,sp_index) 
        ax.imshow(new_set)    
        ax.set_title('Observation no.: ' + str(indexer)
        + '\n Predicted value: ' + str(y_label))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        sp_index += 1
        
    plt.tight_layout()
    plt.show()
    
g = sns.countplot(y_train)
y_train.value_counts()
training_set.isnull().any().describe()


#### Normalization of the data for faster CNN convergence ####

training_set  = training_set / 255.0
test_set = test_set / 255.0


#### Reshaping the 'images' and temporarily deleting 'id' from test_set####

del test_set['id']

training_set = training_set.values.reshape(-1,28,28,1)
test_set = test_set.values.reshape(-1,28,28,1)

#### Encoding y_labels to vectors ####

y_train = to_categorical(y_train, num_classes = 10)

#### Splitting data to the training/test set ####

random_seed = 10
X_train, X_val, Y_train, Y_val = train_test_split(training_set, y_train, test_size = 0.1, random_state=random_seed)

#### ANN Definition ####

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#### Define optimizer, metrics and loss function ####

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#### Define Learning Rate

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 1 
batch_size = 86

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10, 
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False,
        vertical_flip=False)  

datagen.fit(X_train)


history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)