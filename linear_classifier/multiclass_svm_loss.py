#!/usr/bin/python3

# THIS SCRIPT PROVIDES FUNTIONS THAT IMPLEMENT THE MULTI-CLASS SVM LOSS
# L()- FULL MATRIX IMPLEMENTATION
# L_vec()- VECTORIZED IMPLEMENTATION


import numpy as np


def L(X, Y, W, *, delta=1, alpha=1):
    """
        Q- no of classes
        N- dimension of input data
        P- no of input data

        W- QxN weighting matrix
        X- NxP data matrix
        Y- P vector of correct class
        delta- difference margin for computing svm loss
        alpha- regulatization coefficeint
    """

    # compute scores QxP- each column represents scores for one input data
    S= W.dot(X)

    # compute losses for each input data
    L_i= np.zeros(S.shape[1])

    # range that goes over each column(input data) in the scores matrix
    s_range= range(S.shape[1])
    # iterate over each column in S
    for idx in s_range:
        L_i_vec= np.fmax(0, S[:, idx] - S[Y[idx],idx] + delta)
        # set loss of correct class to zero
        L_i_vec[Y[idx]]= 0
        # sum all the losses for each class in the current input data sample
        L_i[idx]= np.sum(L_i_vec)

    # compute data loss
    data_loss= (1/X.shape[0])*np.sum(L_i)

    # compute regularization loss
    reg_loss= np.sum(W*W)

    return (data_loss+reg_loss)



def L_vec(x, yi, W, *, delta=1, alpha=1):
    """
        vectorized implementation
        x- N vector of input data
        yi- scalar of correct class index
        W- QxN matrix of weights
    """

    # compute class scores
    s= W.dot(x)

    # compute vector of losses
    li= np.fmax(0, s + (1-s[yi]))
    li[yi]= 0
    # compute data loss
    l_data= np.sum(li)

    # compute regularization loss
    l_reg= np.sum(W*W)

    # compute total loss
    return l_data