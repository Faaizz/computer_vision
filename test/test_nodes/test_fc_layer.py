import unittest
import sys
import numpy as np

from src.nodes.fc_layer import FCLayer
from src.utils.numeric_gradient import evaluate_gradient

class TestFCLayer(unittest.TestCase):

    def setUp(self):
        W= np.array([[0.2,-0.5,0.1,2.0],[1.5,1.3,2.1,0.0],[0.0,0.25,0.2,-0.3]])
        b= np.array([[1.1,3.2,-1.2]])
        W_b= np.append(W, b.T, axis=1)
        self.model= FCLayer(W_b)

    def test_forward(self):
        data= np.array([[56,231,24,2],[56,231,24,2]]).T

        result= self.model.forward(data)
        exp_result= np.array([[-96.8,437.9,60.75],[-96.8,437.9,60.75]]).T
        # Assertion
        np.testing.assert_almost_equal(result, exp_result)

    def test_backward(self):
        data= np.array([[56,231,24,2],[0,210,240,20]]).T
        # Run forward pass to store data in FCLayer object
        self.model.forward(data)
        up_grad= np.array(np.random.randn(*(self.model.W_b.shape[0],data.shape[1])))
        # up_grad= np.array(np.ones((self.model.W_b.shape[0],data.shape[1])))
        result= self.model.backward(up_grad)

        def func(x): 
            self.model= FCLayer(x)
            return self.model.forward(data)*up_grad
        exp_result= evaluate_gradient(func, self.model.W_b)

        # Assertion
        # sys.stderr.write(str(result))
        # sys.stderr.write(str(exp_result))
        np.testing.assert_almost_equal(result, exp_result)


        
