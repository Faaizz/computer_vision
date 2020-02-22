#!/usr/bin/python3
import numpy as np
import random

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

    def predict(self, data, *, k=1):
        """ data is N x D where N is the number of samples for which to predict """
        # Setup result array
        result= np.zeros(data.shape[0], dtype=self.labels.dtype)
        # Loop through the data to predict
        for idx in range(data.shape[0]):
            
            # Find L1 distance between each col of training data and current prediction sample (broadcasting)
            l1_dis= np.abs(self.data-data[idx,:])
            # Sum up the col distances on each row
            l1_dis= np.sum(l1_dis, axis=1) 
            # L2
            # l2_dis= np.sqrt(np.sum(np.square(self.data-data[idx,:]), axis=1))   
            # Select distance to use
            dis= l1_dis
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
            
            

        # Return predicted labels
        return result





