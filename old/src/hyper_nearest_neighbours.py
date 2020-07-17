#!/usr/bin/python3

import numpy as np

from nearest_neighbour.nearest_neighbour import NearestNeighbour
import utils.cifar_10_data as cifar_10


# Get 1 batch of training data from cifar_10
training= cifar_10.get_training_data(no_of_batches=5)

# validation data
val_data= np.array(training["data"][:, :1000])
val_labels= np.array(training["labels"][:1000])
# Training data
data= np.array(training["data"][:, 1000:])
labels= np.array(training["labels"][1000:])

# Create model
model= NearestNeighbour()

# Train model
print('Training model...')
model.train(data, labels)
print('Model trained.')

# Find the best 'k'
print('Finding best parameters...')
k_range= [1, 2, 3, 4, 5, 10, 50]
best_params= model.get_hyperparams(k_range, val_data, val_labels)
k= k_range[best_params[0]]
metric= best_params[1]
print('Best parameters found.')
# Print best parameters
print('best k is: ')
print(k)
print('\nbest metric is: ')
print(metric)

# Log "best" parameters to a json file at "./nearest_neighbour/params.json"
import json

with open("nearest_neighbour/params.json", "w") as f:
    json.dump({"k": k, "metric": metric}, f)



