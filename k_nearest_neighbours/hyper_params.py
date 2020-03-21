#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
#from nearest_neighbour import NearestNeighbour

# THIS SCRIPT PROVIDES METHODS THAT TESTS A RANGE OF 'k's AGAINST L1 AND L2 DISTANCE METRICS AND RETURN THE MOST ACCURATE COMBINATION

def get_hyperparams(k_range, val_data, val_labels, model):
    """ k_range- a list of k's to try out.
        val_data- an NxD array with each row a validation data
        val_labels- a 1-dimensional array of corresponding data labels
        model- trained model
    """

    l1_scores= np.zeros(k_range.shape[0])
    l2_scores= np.zeros(k_range.shape[0])

    for idx in range(k_range.shape[0]):
        # print ('k iteration number: ') 
        # print(idx)
        # compute L1 accuracy
        val_pred_l1= model.predict(val_data, k=k_range[idx], metric=1)
        l1_scores[idx]= np.mean(val_pred_l1 == val_labels)
        # compute L2 accuracy
        val_pred_l2= model.predict(val_data, k=k_range[idx], metric=2)
        l2_scores[idx]= np.mean(val_pred_l2 == val_labels)


    # Plot the accuracies
    plt.plot(k_range, l1_scores, '*')
    plt.plot(k_range, l2_scores, '*')
    plt.legend(['L1 accuracies', 'L2 accuracies'])
    plt.show()

    # find index of k with best L1 accuracy
    k_l1= np.argmax(l1_scores)
    # find index of k with the best l2 score
    k_l2= np.argmax(l2_scores)

    # compare L1 and L2 scores for the highest accuracy
    if np.amax(l1_scores) > np.amax(l2_scores):
        # 1 returned signifies that L1 metric performed better
        return (k_l1, 1)
    else:
        # 2 returned signifies that L2 metric performed better
        return (k_l2, 2)
