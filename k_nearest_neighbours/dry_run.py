#!/usr/bin/python3

# THIS SCRIPT TESTS THE ALGORITHM ON AN ARRAY OF ZEROS AND ONES

from nearest_neighbour import NearestNeighbour
import numpy as np

# Demo Training Data
data= np.array([np.zeros((4,)), np.ones((4,)), np.zeros(4,), np.zeros(4,), np.ones(4,)])

# Demo Training Labels
labels_tag= np.array(['zeros', 'ones'])
labels= np.array([0, 1, 0, 0, 1])

# Create Model
model= NearestNeighbour()
# Train model
model.train(data, labels)

# Test Model
accuracy= model.test(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]), np.array([0,1]), k=2, metric=2)
# Print result
print("Accuracy is: %.2f%%" % (accuracy*100))