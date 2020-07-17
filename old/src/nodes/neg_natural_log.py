# THIS SCRIPT PROVIDES THE NEGATIVE NATURAL LOG FUNCTION AS A NODE IN A COMPUTATION GRAPH

import numpy as np
import sys

class NegNatLog:

    def __init__(self):
        pass

    def forward(self, softmax):
        """
            Compute forward propagation
            - softmax: ndarray(p,). Vector of Softmax output
        """
        # Keep input for backprop
        self.softmax= softmax
        # Calculate and return elementwise natural logarithm
        return -np.log(softmax)

    def backward(self, up_grad):
        """
            Compute local gradient and backprop
            - up_grad: ndarray(m,). Upstream gradient
        """
        softmax= self.softmax
        # Local gradient
        loc_grad= -1/softmax
        # sys.stderr.write(str((loc_grad*up_grad).shape))
        return loc_grad*up_grad