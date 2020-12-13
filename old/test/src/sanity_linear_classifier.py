#!/usr/bin/python3

import numpy as np
from linear_classifier.linear_classifier import LinearClassifier
from utils import cifar_10_data as cifar_10
from utils import local_img_data
import random


# Create model
model= LinearClassifier()

# Get training data
train= cifar_10.get_training_data()
train_data, train_labels= train.values()
train_data= np.array(train_data[:, :5], dtype=np.float64)
train_labels= train_labels[:5]
# Zero centering
train_data-= np.mean(train_data)
train_data /= np.std(train_data)
# train_data= np.array([[-2,-1],[0,2],[4,-4],[1,6]])
# train_labels= np.array([0,2])
# Train model with minimal data so as to overfit
print('Training model...')
model.train(train_data, train_labels)
print('Model trained.')