
import unittest, sys
import numpy as np

from src.nodes.multiclass_svm_loss import MulticlassSVMLoss
from src.utils.numeric_gradient import evaluate_gradient

class TestMulticlassSVMLoss(unittest.TestCase):

    def setUp(self):
        self.model= MulticlassSVMLoss()

    def test_forward(self):
        scores= np.array([[-2.85,0.86,0.28],[-2.85,1.86,0.28]]).T
        labels= np.array([2,2])
        result= self.model.forward(scores, labels)
        # Assertion
        exp_result= np.array([1.58,2.58])
        np.testing.assert_almost_equal(result, exp_result)

    def test_backward(self):
        scores= np.array([[-2.85,0.86,0.28],[-2.85,1.86,0.28]]).T
        labels= np.array([2,2])
        forward= self.model.forward(scores, labels)
        up_grad= np.random.randn(*forward.shape)
        result= self.model.backward(up_grad)

        func= lambda x: self.model.forward(x,labels)
        exp_result= evaluate_gradient(func, scores)*up_grad
        sys.stderr.write(str(result.shape))
        # Assertion
        np.testing.assert_almost_equal(result, exp_result)