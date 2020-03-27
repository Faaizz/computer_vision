#!/usr/bin/python3

# THIS SCRIPT PROVIDES FUNCTIONS THAT COMPUTES THE NUMERIC GRADIENT OF A FUNTION

import numpy as np

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
        # move in W-h
        W[idx]= old - h
        # evaluate L(W-h)
        L_minus= L(W)
        # evaluate the central difference (L(W+h)-L(W-h))/2h
        dW[idx]= (L_plus-L_minus)/(2*h)

        # revert back to original W
        W[idx]= old
        # move to next element
        it.iternext()

    return dW



