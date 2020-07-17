#!/usr/bin/python3
import numpy as np
import random
import sys

class NearestNeighbour:

    def __init__(self):
        pass

    def train(self, data, labels):
        """ data is N x D while labels is 1-dim with length N.
        labels are integers which should represent the indices of the respective labels
        """
        # Simply remember the training data
        self.data= data
        self.labels= labels

    def predict(self, data, *, k=1, metric=1):
        """ data is N x D where N is the number of samples for which to predict
            metric- 1 selects L1 distance metric. L2 is used otherwise
        """
        # Setup result array
        result= np.zeros(data.shape[1], dtype=self.labels.dtype)

        # Loop through the data to predict
        for idx in range(data.shape[1]):
            
            if metric==1:
                # Find L1 distance between each col of training data and current prediction sample (broadcasting) and
                # Sum up the col distances on each row
                dis= np.sum(np.abs(self.data.T-data.T[idx,:]), axis=1)
            else:
                # L2
                dis= np.sqrt(np.sum(np.square(self.data.T-data.T[idx,:]), axis=1))   

            # Find the data with lowest distance
            top_match= self.labels[np.argmin(dis)]       
            # Identify the first 'k' rows with the smallest distance
            # argsort returns the indices of the sorted array, doesn't actually perform sort
            winners= np.argsort(dis)[:k]
            winners_labels= self.labels[winners]
            # Count the label frequency and pick the index with the the highest
            # bincount returns the frequency of each array element as it's index
            # eg: frequency of 7 is returned as bin_result[6]
            winner_label_bin_count= np.bincount(winners_labels)
            win= np.array(np.where(winner_label_bin_count==np.max(winner_label_bin_count)))
            win= win.flatten()
            # print(winners_labels)
            # print(win)
            # If there are multiple winners and the data with lowest distance is amongst them, pick this
            if(win.shape[0]>1):
                if top_match in win:
                    result[idx]= top_match
                else:
                    # Pick random winner
                    winner= win[random.randrange(0, win.shape[0])]
            else:
                # Select the single winner
                result[idx]= win[0]
                # print(top_match)
                # print(result[idx])
            
            # print ('data iteration number: ') 
            # print(idx)
            

        # Return predicted labels
        return result


    def get_hyperparams(self, k_range, val_data, val_labels):
        """ k_range- a list of k's to try out (regular python list, not a numpy array).
            val_data- an NxD array with each row a validation data
            val_labels- a 1-dimensional array of corresponding data labels
        """

        l1_scores= np.zeros(len(k_range))
        l2_scores= np.zeros(len(k_range))

        for idx in range(len(k_range)):
            # print ('k iteration number: ') 
            # print(idx)
            # compute L1 accuracy
            val_pred_l1= self.predict(val_data, k=k_range[idx], metric=1)
            # print(val_labels == val_pred_l1)
            l1_scores[idx]= np.mean(val_pred_l1 == val_labels)
            val_pred_l2= self.predict(val_data, k=k_range[idx], metric=2)
            l2_scores[idx]= np.mean(val_pred_l2 == val_labels)


        # Plot the accuracies
        import matplotlib.pyplot as plt
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




