#!/usr/bin/python3

# THIS SCRIPT IMPORTS TRAINING DATA FROM THE CIFAR-10 DATASET AND TESTS THE ALGORITHM ON 
# A 32x32 cat.png IMAGE and A 32x32 airplane.jpg IMAGE AND ALSO ON TEST DATASET FROM CIFAR-10

import numpy as np
from nearest_neighbour.nearest_neighbour import NearestNeighbour
from utils import cifar_10_data as cifar_10
from utils import local_img_data
import random



# Create model
model= NearestNeighbour()

# Get training data
train= cifar_10.get_training_data()
train_data, train_labels= train.values()
# Train model
print('Training model...')
model.train(train_data, train_labels)
print('Model trained.')

# Load hyperparameters
import json
with open("nearest_neighbour/params.json", "r") as f:
    hyperparams= json.load(f)

k, metric= hyperparams.values()

# Testing
def test_cifar():
    print("Testing on CIFAR-10 dataset...")
    # Get testing data
    testing= cifar_10.get_test_data()
    test_data, test_labels= testing.values()
    # Predict
    results= model.predict(test_data, k=k, metric=metric)
    accuracy= np.mean(results == test_labels)*100
    print("CIFAR-10 testing completed.")
    # Print Results
    print("{0:30}: {1:>5}%".format("CIFAR-10 Accuracy", accuracy))

def test_local():
    # Predict Test Images sourced from Google
    print("Testing local images...")
    testing_local= local_img_data.get_test_data()
    local_data, local_labels= testing_local.values()
    results_local= model.predict(local_data, k=k, metric=metric)
    local_accuracy= np.mean(results_local == local_labels)*100
    print(results_local)
    print(local_labels)
    print("Local testing completed")
    # Print Results
    print("{0:30}: {1:>5}%".format("Local Images Accuracy", local_accuracy))

# test_cifar()
test_local()

