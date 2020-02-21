#!/usr/bin/python3

# THIS SCRIPT IMPORTS TRAINING DATA FROM THE CIFAR-10 DATASET AND TESTS THE ALGORITHM ON 
# A 32x32 cat.png IMAGE and A 32x32 airplane.jpg IMAGE AND ALSO ON TEST DATASET FROM CIFAR-10

import numpy as np
from nearest_neighbour import NearestNeighbour
from matplotlib.pyplot import imread

def unpickle(file):
    """Unpickles a CIFAR-10 dataset file"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Get training dataset
dataset= unpickle('../cifar-10-batches-py/data_batch_1');
key_list= list(dataset)
data_key= key_list[2]
labels_key= key_list[1]

# Create model
model= NearestNeighbour()

# Train model
model.train(np.array(dataset[data_key]), np.array(dataset[labels_key]))

# Import Test Image
cat_img = np.uint8(imread('./img/cat.jpg'))
cat_img= cat_img.flatten()
airplane_img = np.uint8(imread('./img/airplane.jpg'))
airplane_img= airplane_img.flatten()
 
# Predict Test Images sourced from Google
results= model.predict(np.array([cat_img, airplane_img]))

# Print Results

# Unpickle label names
label_names_data= unpickle('../cifar-10-batches-py/batches.meta')
label_key_list= list(label_names_data)
label_names= label_names_data[label_key_list[1]]

print('Test images from Google')

print('cat image is predicted as a: ' )
print(label_names[results[0]])
print("\n")
print('airplane image is predicted as a: ' )
print(label_names[results[1]])

# Predict Test images from dataset
test_dataset= unpickle('../cifar-10-batches-py/test_batch')
test_key_list= list(test_dataset)
test_labels= test_dataset[test_key_list[1]]
test_data= test_dataset[test_key_list[2]]
first_cat_index= test_labels.index(3)
first_airplane_index= test_labels.index(0)
first_cat= test_data[first_cat_index]
first_airplane= test_data[first_airplane_index]

# Condition data for model
cat_img = np.uint8(first_cat)
airplane_img = np.uint8(first_airplane)
 
# Predict Test Images sourced from Google
results= model.predict(np.array([cat_img, airplane_img]))

print('\n\n\n')
print('Test images from dataset')

print('cat image is predicted as a: ' )
print(label_names[results[0]])
print("\n\n")
print('airplane image is predicted as a: ' )
print(label_names[results[1]])