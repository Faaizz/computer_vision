#!/usr/bin/python3

# THIS SCRIPT PROVIDES FUNCTIONS THAT COMPUTES THE NUMERIC GRADIENT OF A FUNTION

import numpy as np
import sys

def evaluate_gradient(L, W, *, h=.00001):
    """
        P- number of classes
        N- dimention of input data

        L- A (loss) function that takes a single argument
        W- PxN weighting matrix

        h- step
    """

    # evaluate current function value
    L_W= L(W)
    # sys.stderr.write(str(L_W))

    # create gradient matrix
    dW= np.zeros(W.shape)

    # temp W matrix
    W_temp= np.zeros(W.shape)

    # create iterator to iterate over elements of W
    it= np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        # get current index
        idx= it.multi_index

        old= W[idx]
        # move in the direction of the current element
        W[idx]= old + h
        # evaluate L(W+h)
        L_plus= L(W)
        # sys.stderr.write(str(L_plus))
        # move in W-h
        W[idx]= old - h
        # evaluate L(W-h)
        L_minus= L(W)
        # sys.stderr.write(str(old-h))
        # evaluate the central difference (L(W+h)-L(W-h))/2h
        # Check if function output is a vector
        # if (type(L_plus).__module__== np.__name__) and len(L_plus.shape) > 0 and L_plus.shape[0] > 1:

        diff= np.sum(L_plus-L_minus)
        # diff= L_plus-L_minus

        dW[idx]= (diff)/(2*h)
        # revert back to original W
        W[idx]= old
        # move to next element
        it.iternext()


    # sys.stderr.write(str(dW))
    return dW



