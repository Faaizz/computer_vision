
# THIS SCRIPT PROVIDES THE MULTI-CLASS SVM LOSS FUNCTION AS A NODE IN A COMPUTATION GRAPH


import numpy as np
import sys

class MulticlassSVMLoss:

    def __init__(self):
        pass

    def forward(self, scores, labels, delta=1, alpha=1):
        """
            -scores: ndarray(output_dim, no_of_inputs). Matrix of scores. Each column represent the score of an input sample
            -labels: ndarray(no_of_inputs,). Vector 
            -delta: difference margin for computing svm loss
        """
        # Save scores for backprop
        self.scores= scores
        self.labels= labels
        L_i_mat= np.fmax(0, (scores+1)-scores[labels, range(scores.shape[1])])
        L_i_mat[labels,range(scores.shape[1])]= 0
        self.L_i_mat= L_i_mat
        L_i= np.sum(L_i_mat, axis=0)
        return L_i

    def backward(self, up_grad):
        """
            -scores: ndarray(output_dim,). Upstream gradient.
        """
        loc_grad= self.L_i_mat/self.L_i_mat
        loc_grad[self.labels, range(loc_grad.shape[1])]= -1
        # Replace nan with zero
        loc_grad= np.nan_to_num(loc_grad)
        return loc_grad*up_grad
