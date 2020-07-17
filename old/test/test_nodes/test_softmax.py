import unittest
import numpy as np

import sys

from src.nodes.softmax import Softmax
from src.utils.numeric_gradient import evaluate_gradient

class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.model= Softmax()

    def test_forward_vec(self):
        s= np.zeros(10)
        result= self.model.forward_vec(s, 1)
        # Assertion
        self.assertEqual(result, 0.1)

        # Make s[1]= 100 such that /exp(s[1]) >> /exp(s[j]) for j!=1
        s[1]= 100
        result= self.model.forward_vec(s, 1)
        # Assertion
        self.assertEqual(result, 1)

    def test_forward(self):
        s= np.zeros((10, 100))
        win_idx= np.ones(100, dtype="int64")
        result= self.model.forward(s, win_idx)
        # Assertion
        res_exp= np.zeros(100)+0.1
        np.testing.assert_almost_equal(result, res_exp)

    def test_backward_vec(self):
        s= np.zeros(10)
        s[1]= 10
        result= self.model.forward_vec(s, 1)
        
        # Compare with Numeric Gradient 
        up_grad= 1
        grad_a= self.model.backward_vec(up_grad)
        soft= lambda x: self.model.forward_vec(x, 1)
        grad_num= evaluate_gradient(soft, s)
        np.testing.assert_almost_equal(grad_a, grad_num)

    def test_backward(self):
        s= np.zeros((10, 100))
        s[1,:]= 10
        win_idx= np.ones(100, dtype="int64")
        result= self.model.forward(s, win_idx)
        up_grad= np.random.randn(*result.shape)

        # Compare with Numeric Gradient 
        soft= lambda x: self.model.forward(x, win_idx)*up_grad
        grad_a= self.model.backward(up_grad)
        grad_num= evaluate_gradient(soft, s)
        np.testing.assert_almost_equal(grad_a, grad_num)

