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

# Create model
model= NearestNeighbour()

# Train model
model.train(np.array(data), np.array(labels))

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