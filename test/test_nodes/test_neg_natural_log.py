import unittest
import sys
import numpy as np

from src.nodes.neg_natural_log import NegNatLog
from src.utils.numeric_gradient import evaluate_gradient

class TestNatLog(unittest.TestCase):

    def setUp(self):
        self.model= NegNatLog()

    def test_forward(self):
        data= np.abs(np.random.randn(10))
        result= self.model.forward(data)        
        # Assertion
        np.testing.assert_almost_equal(result, -np.log(data))

    def test_backward(self):
        data= np.abs(np.random.randn(10))
        # data= np.array([0.5, 0.35, 0.0015])
        forward= self.model.forward(data)
        up_grad= np.random.randn(forward.shape[0])
        result= self.model.backward(up_grad)
        # Assertion
        np.testing.assert_almost_equal(result, (-1/data)*up_grad)
        # Compare with Numeric Gradient
        func= lambda x: self.model.forward(x)*up_grad
        grad_num= evaluate_gradient(self.model.forward, data, h=1e-3)
        np.testing.assert_almost_equal(result, grad_num, decimal=3)