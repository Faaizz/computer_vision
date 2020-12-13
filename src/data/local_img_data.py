
# THIS SCRIPT SUPPLIES TRAINING, AND TESTING DATA FROM CIFAR-10 DATASET
# THE DATASET IS LOCATED IN "/cifar-10-batches-py" FOLDER IN THE ROT DIRECTORY OF THE PROJECT

import numpy as np
import os
import sys
from matplotlib.pyplot import imread

# Get current path
root_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root path
root_dir= root_dir.replace("/src/utils", "")

# Import Test Images
local_img_1 = np.uint8(imread(root_dir + '/img/airplane.jpeg'))
local_img_1= local_img_1.flatten(order='F')

local_img_2 = np.uint8(imread(root_dir + '/img/automobile.jpg'))
local_img_2= local_img_2.flatten(order='F')

local_img_3 = np.uint8(imread(root_dir + '/img/bird.jpg'))
local_img_3= local_img_3.flatten(order='F')

local_img_4 = np.uint8(imread(root_dir + '/img/cat.jpeg'))
local_img_4= local_img_4.flatten(order='F')

local_img_5 = np.uint8(imread(root_dir + '/img/truck.jpg'))
local_img_5= local_img_5.flatten(order='F')

local_img_6 = np.uint8(imread(root_dir + '/img/ship.jpeg'))
local_img_6= local_img_6.flatten(order='F')

data= np.array([local_img_1, local_img_2, local_img_3, local_img_4, local_img_5, local_img_6])
labels= np.array([0, 1, 2, 3, 9, 8])

def get_test_data():
    """
        Returns training dataset(s) as dictionary of numpy-array values in which column of 'data' represents a single data sample
        Returns the labels as strings
    """
    
    return {"data":data.T,"labels":labels}

