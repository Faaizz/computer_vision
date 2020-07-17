import unittest
import random
import sys

import numpy as np

from src.nearest_neighbour.nearest_neighbour import NearestNeighbour
import src.utils.cifar_10_data as cifar_10

class TestNearestNeightbour(unittest.TestCase):


    def setUp(self):
        """
            Runs before each test method
        """
        self.model= NearestNeighbour()
        self.training= cifar_10.get_training_data()

    
    def test_train_L1_and_predict(self):
        """
            Test training with L1 distance metric and k=1
        """
        
        # Matrix of training data with each column representing a trianing data
        train_data= self.training["data"]
        train_labels= self.training["labels"]

        # Train model
        self.model.train(train_data, train_labels)

        # Predict known samples selected randomly
        to_predict_1= random.randrange(train_data.shape[1])
        to_predict_2= random.randrange(train_data.shape[1])
        to_predict_data= train_data[:,[to_predict_1, to_predict_2]]
        # sys.stderr.write(str(to_predict_data))
        to_predict_labels= train_labels[to_predict_1]
        to_predict_labels= np.append(to_predict_labels, train_labels[to_predict_2])

        results= self.model.predict(to_predict_data)

        # loop through array and compare results
        for idx in range(to_predict_labels.shape[0]):
            # Assertion
            self.assertEqual(results[idx], to_predict_labels[idx])


    def test_train_L2_and_accuracy(self):
        """
            Test accuracy with L2 distance metric and k=3
        """
        
        # Matrix of training data with each column representing a trianing data
        train_data= np.array([[1,1,1,1], [0,0,0,0], [4,5,5,5], [4,5,5,5], [5,5,5,5]])
        train_data= train_data.T
        train_labels= np.array([1,0,4,4,5])
        # Train model
        self.model.train(train_data, train_labels) 
        # Predict result
        to_predict_data= np.array([[5,5,5,5], [1,1,1,1]]).T
        to_predict_label= np.array([4, 1])
        result= self.model.predict(to_predict_data, k=3, metric=2)

        # Assertion
        for idx in range(result.shape[0]):
            self.assertEqual(result[idx], to_predict_label[idx])

    
    def test_get_hyperparams(self):
        """
            Test determination of hyperparameters
        """
        # Matrix of training data with each column representing a trianing data
        train_data= np.array([[1,1,1,1], [0,0,0,0], [4,4,4,4], [5,5,5,5]])
        train_data= train_data.T
        train_labels= np.array([1,0,4,5])
        # Train model
        self.model.train(train_data, train_labels) 

        # Matrix of validation data with each column representing a trianing data
        val_data= np.array([[1,0,1,1], [1,0,0,0], [5,4,4,4], [5,5,5,5]])
        val_data= val_data.T
        val_labels= np.array([0,0,4,6])

        params= self.model.get_hyperparams([1,2,3,4,5,10,50], val_data, val_labels)

        sys.stderr.write(str(params))



    

