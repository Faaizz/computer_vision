#!/usr/bin/python3

# THIS SCRIPT IMPORTS TRAINING DATA FROM THE CIFAR-10 DATASET AND TESTS THE ALGORITHM ON 
# A 32x32 cat.png IMAGE and A 32x32 airplane.jpg IMAGE AND ALSO ON TEST DATASET FROM CIFAR-10

import numpy as np
from nearest_neighbour import NearestNeighbour
from matplotlib.pyplot import imread
import random
from hyper_params import get_hyperparams

def unpickle(file):
    """Unpickles a CIFAR-10 dataset file"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Import Test Images
local_img_1 = np.uint8(imread('./img/airplane.jpeg'))
local_img_1= local_img_1.flatten(order='F')

local_img_2 = np.uint8(imread('./img/automobile.jpg'))
local_img_2= local_img_2.flatten(order='F')

local_img_3 = np.uint8(imread('./img/bird.jpg'))
local_img_3= local_img_3.flatten(order='F')

local_img_4 = np.uint8(imread('./img/cat.jpeg'))
local_img_4= local_img_4.flatten(order='F')

local_img_5 = np.uint8(imread('./img/truck.jpg'))
local_img_5= local_img_5.flatten(order='F')

local_img_6 = np.uint8(imread('./img/ship.jpeg'))
local_img_6= local_img_6.flatten(order='F')

# Get training dataset
dataset1= unpickle('../cifar-10-batches-py/data_batch_1');
dataset2= unpickle('../cifar-10-batches-py/data_batch_2');
dataset3= unpickle('../cifar-10-batches-py/data_batch_3');
dataset4= unpickle('../cifar-10-batches-py/data_batch_4');
dataset5= unpickle('../cifar-10-batches-py/data_batch_5');

key_list= list(dataset1)
data_key= key_list[2]
labels_key= key_list[1]

data= dataset1[data_key]
data= np.append(data,dataset2[data_key],axis=0)
data= np.append(data,dataset3[data_key],axis=0)
data= np.append(data,dataset4[data_key],axis=0)
data= np.append(data,dataset5[data_key],axis=0)

labels= dataset1[labels_key]
labels= np.append(labels,dataset2[labels_key],axis=0)
labels= np.append(labels,dataset3[labels_key],axis=0)
labels= np.append(labels,dataset4[labels_key],axis=0)
labels= np.append(labels,dataset5[labels_key],axis=0)

# split validation and training data
# validation
val_data= data[:1000, :]
val_labels= labels[:1000]
# training
data= data[1000:, :]
labels= labels[1000:]


# Create model
model= NearestNeighbour()

# Train model
print('Training model...')
model.train(np.array(data), np.array(labels))
print('Model trained.')

# Find the best 'k'
print('Finding best parameters...')
k_range= np.array([1, 2, 3, 4, 5, 10, 50])
best_params= get_hyperparams(k_range, val_data, val_labels, model)
k= k_range[best_params[0]]
metric= best_params[1]
print('Best parameters found.')
# Print best parameters
print('best k is: ')
print(k)
print('\nbest metric is: ')
print(metric)

# Predict Test Images sourced from Google
results= model.predict(np.array([local_img_1, local_img_2, local_img_3, local_img_4, local_img_5, local_img_6]), k=k, metric=metric)

# Print Results

# Unpickle label names
label_names_data= unpickle('../cifar-10-batches-py/batches.meta')
label_key_list= list(label_names_data)
label_names= label_names_data[label_key_list[1]]
print(label_names)
print('Test images from Google')
print('airplane image is predicted as a: ' )
print(label_names[results[0]])
print('automobile image is predicted as a: ' )
print(label_names[results[1]])
print('bird image is predicted as a: ' )
print(label_names[results[2]])
print('cat image is predicted as a: ' )
print(label_names[results[3]])
print('truck image is predicted as a: ' )
print(label_names[results[4]])
print('horse image is predicted as a: ' )
print(label_names[results[5]])

# Test model
test_dataset= unpickle('../cifar-10-batches-py/test_batch')
test_key_list= list(test_dataset)
test_labels= test_dataset[test_key_list[1]]
test_data= test_dataset[test_key_list[2]]
# Select 2 random peices from test data
# test_1_idx= test_labels[random.randrange(0, test_data.shape[0]/2)]
# test_2_idx= test_labels[random.randrange(test_data.shape[0]/2, test_data.shape[0])]

# Condition data for model
# test_img_1 = np.uint8(test_data[test_1_idx])
# test_img_2 = np.uint8(test_data[test_2_idx])


print("Testing model...")
# Predict Test Images from CIFAR-10 test set
accuracy= model.test(test_data, test_labels, k=k, metric=metric)
# Print result
print("Accuracy is %.2f%%" %(accuracy*100))

