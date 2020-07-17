import unittest
import sys
import numpy as np

import src.utils.cifar_10_data as cifar_10

class TestCifar10Data(unittest.TestCase):

    def test_single_data_sample_length(self):
        # Request single batch
        res= cifar_10.get_training_data()
        # get data
        data= res["data"]
        # verify that each column represents a data sample of 3072 entries
        self.assertEqual(data.shape[0], 3072)

    @unittest.expectedFailure
    def test_invalid_multiple_training_batches_import(self):
        # Request 15 batches
        success= True
        try:
            cifar_10.get_training_data(no_of_batches=15)
        except Exception as e:
            success= False
        
        # assertion
        self.assertTrue(success)

    def test_multiple_training_batches_import(self):
        # Request 5 training batches
        res= cifar_10.get_training_data(no_of_batches=5)
        # get data
        data= res["data"]
        # assert that all 50000 data samples were obtained
        self.assertEqual(data.shape[1], 50000)
        # get labels
        labels= res["labels"]
        # assert that all 50000 labels were obtained
        self.assertEqual(labels.shape[0], 50000)

