import unittest
import sys

import numpy as np

# from src.linear_classifier.linear_classifier import LinearClassifier
from src.nodes.softmax import Softmax
from src.nodes.neg_natural_log import NegNatLog
from src.nodes.fc_layer import FCLayer
from src.nodes.multiclass_svm_loss import MulticlassSVMLoss

from src.utils.numeric_gradient import evaluate_gradient

class TestLinearClassifier(unittest.TestCase):

    def setUp(self):
        W= np.array([[0.01,-0.05,0.1,0.05],[0.7,0.2,0.05,0.16],[0.0,0-0.45,-0.2,0.03]])
        b= np.array([[0.0,0.2,-0.3]])
        self.W_b= np.append(W, b.T, axis=1)

    def test_forward_pass(self):
        data= np.array([[-15,22,-44,56],[-15,22,-44,56]]).T
        labels= np.array([2, 2])
        softmax= Softmax()
        neg_nat_log= NegNatLog()
        fc_layer= FCLayer(self.W_b)

        # Loss per data sample
        L_i= neg_nat_log.forward(softmax.forward(fc_layer.forward(data, W_b=self.W_b), labels))
        # sys.stderr.write(str(L_i))
        # Total Loss
        L= np.sum(L_i)/L_i.shape[0]

        # Assertion
        np.testing.assert_almost_equal(L, 1.04, decimal=2)

    def test_backward(self):
        data= np.array([[-1,29,14,-60],[-15,22,-44,56]]).T
        labels= np.array([2, 0])
        # Randomize weights
        self.W_b= np.random.randn(*self.W_b.shape) * 0.01
        softmax= Softmax()
        neg_nat_log= NegNatLog()
        fc_layer= FCLayer(self.W_b)
        for idx in range(1):
            # Analytic gradient
            # Forward pass
            scores= fc_layer.forward(data, W_b=self.W_b)
            smx= softmax.forward(scores, labels)
            # sys.stderr.write(str(scores.shape))
            L_i= neg_nat_log.forward(smx)
            L= np.sum(L_i)/L_i.shape[0]
            sys.stderr.write(str(L)+ "\n")
            # Backprop
            up_grad= np.ones(data.shape[1])/data.shape[1]
            ana_grad= fc_layer.backward(softmax.backward(neg_nat_log.backward(up_grad)))

            def func(W_b):
                # Loss per data sample
                L_i= neg_nat_log.forward(softmax.forward(fc_layer.forward(data, W_b=W_b), labels))
                # Total Loss
                return np.sum(L_i)/L_i.shape[0]

            # Numeric gradient
            num_grad= evaluate_gradient(func, self.W_b)

            # Assertion
            np.testing.assert_almost_equal(ana_grad, num_grad, decimal=4)
            # Update gradient
            self.W_b += -ana_grad*1e-4

    def test_backward_multiclass_svm(self):
        data= np.array([[-1,29,14,-60],[-15,22,-44,56]]).T
        labels= np.array([2, 0])
        # Randomize weights
        self.W_b= np.random.randn(*self.W_b.shape) * 0.01
        
        multi_svm_loss= MulticlassSVMLoss()
        fc_layer= FCLayer(self.W_b)
        # Forward pass
        L_i= multi_svm_loss.forward(fc_layer.forward(data), labels)
        Loss= np.sum(L_i)
        # Backprop
        up_grad= np.random.randn(*L_i.shape)
        result= fc_layer.backward(multi_svm_loss.backward(up_grad))
        # Numeric gradient
        def func(x):
            fc_layer= FCLayer(x)
            return multi_svm_loss.forward(fc_layer.forward(data), labels)*up_grad
        exp_result= evaluate_gradient(func, self.W_b)
        # Assertion
        np.testing.assert_almost_equal(result, exp_result)
        

