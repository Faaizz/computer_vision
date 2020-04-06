
# THIS SCRIPT PROVIDES THE SOFTMAX FUNCTION AS A NODE IN A COMPUTATION GRAPH

import numpy as np
import sys

class Softmax:

    def forward_vec(self, scores, win_idx):
        """
            Compute forward propagation.
            - scores: ndarray(p,). Vector of class scores
            - win_idx: int. Index of ground truth 
        """
        # Keep scores for backprop
        self.scores= scores
        # Keep win_idx for backprop
        self.win_idx= win_idx
        # Softmax
        h= np.exp(scores)
        self.softmax_result= h
        
        return h[win_idx]/np.sum(h)

    def forward(self, scores, win_idx):
        """
            Compute forward propagation.
            - scores: ndarray(p,m). Matrix with class scores as columns.
            - win_idx: ndarray(m,). Vector with ground truth
        """
        # For numerical stability
        # scores -= np.max(scores) 
        # Keep scoes for backprop
        self.scores= scores
        # Keep win_idx for backprop
        self.win_idx= win_idx
        # Softmax
        # h= np.exp(scores)
        # sys.stderr.write(str(h.shape))
        h= np.exp(scores - np.max(scores))
        self.softmax_result= h
        col_idx= np.arange(scores.shape[1])
        ground_truths= h[win_idx, col_idx]
        
        return ground_truths/np.sum(h, axis=0)


    def backward_vec(self, up_grad):
        """
            Compute local gradient and backprop
            - up_grad: ndarray(1). Upstream gradient
        """ 
        h= self.softmax_result
        # Local gradient
        # For j != win_idx
        loc_grad= h * ((-h[self.win_idx])/np.sum(h)**2)
        # For j== win_idx
        loc_grad[self.win_idx]= loc_grad[self.win_idx] / ((-h[self.win_idx])/np.sum(h)**2) * ((np.sum(h)-h[self.win_idx])/np.sum(h)**2)

        return up_grad*loc_grad

    def backward(self, up_grad):
        """
            Compute local gradient and backprop
            - up_grad: ndarray(m,). Upstream gradient
        """
        h= self.softmax_result
        col_idx= np.arange(h.shape[1])
        # Local gradient
        h_sum= np.sum(h, axis=0)
        den= ((-h[self.win_idx, col_idx])/h_sum**2)
        # sys.stderr.write(str(den))
        # For j != win_idx
        loc_grad= np.array(h * den, dtype=np.double)
        # For j== win_idx
        loc_grad[self.win_idx]= loc_grad[self.win_idx] / den * ((h_sum-h[self.win_idx, col_idx])/h_sum**2)

        return loc_grad*up_grad

