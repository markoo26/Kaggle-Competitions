#### Import of the libraries ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, webbrowser, time, random

#### Import the datasets and preparation####

training_set = pd.read_csv("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Kannada-MNIST\\train.csv")
test_set = pd.read_csv("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\Kannada-MNIST\\test.csv")

y_train = training_set["label"]
training_set = training_set.drop(['label'], axis=1)

#### Check_pictures object definition ####
#### Call the object to see either first n or random n pictures from the dataset ####
#### With observation number and predicted label ####

#### Add charts for one label only 
def check_pictures(dataset, labels, no_of_pictures, randomize = False, 
                   y_label = None):

#    dataset = training_set
#    labels = y_train
#    no_of_pictures = 15
#    randomize = False
#    y_label = 5
    
    fig = plt.figure()
    sp_index = 1

    if y_label:
        image_set = dataset[labels == y_label]
    else:
        image_set = dataset 
        
    if randomize:
        pictures_ids = np.random.randint(1, len(image_set), size=(1, no_of_pictures))
    else:
        pictures_ids = np.array(range(0,no_of_pictures))

    for i in np.nditer(pictures_ids):        
        
        indexer = int(i)
        new_set = image_set.iloc[indexer,:].values.reshape(28,28)
        new_title = labels[indexer]
        ax = fig.add_subplot(3,5,sp_index)
        ax.imshow(new_set)    
        ax.set_title('Observation no.: ' + str(indexer)
        + '\n Predicted value: ' + str(new_title))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        sp_index += 1
        
    plt.tight_layout()
    plt.show()

check_pictures(dataset= training_set, labels = y_train,
               no_of_pictures=15, randomize = True, y_label = 3)