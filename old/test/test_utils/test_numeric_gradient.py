import unittest, sys
import numpy as np

from src.utils.numeric_gradient import evaluate_gradient

class TestNumericGradient(unittest.TestCase):

    def test_numeric_gradient(self):
        # Vector valued function
        def vec_valued(X):
            # return np.array([(X[0]*X[1]), (X[0]+X[1])])
            return np.array([(X[0]*X[1]), (X[0]+X[1])], dtype=np.float32)

        # Gradient
        grad= evaluate_gradient(vec_valued, np.array([2,3], dtype=np.float32))
        sys.stderr.write(str(grad))