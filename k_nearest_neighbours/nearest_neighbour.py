#!/usr/bin/python3
import numpy as np

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
            
            # Find L2 distance between each col of training data and current prediction sample (broadcasting)
            l2_dis= np.sqrt(np.square(self.data-data[idx]))
            # Sum up the col distances on each row
            l2_dis_sum= np.sum(l2_dis, axis=1)
            # Identify the first 'k' rows with the smallest distance
            # argsort returns the indices of the sorted array, doesn't actually perform sort
            l2_winners= np.argsort(l2_dis_sum)[:k]
            l2_winners_labels= self.labels[l2_winners]
            # Count the label frequency and pick the index with the the highest
            # bincount returns the frequency of each array element as it's index
            # eg: frequency of 7 is returned as bin_result[6]
            print(l2_winners_labels)
            l2_winner= np.argmax(np.bincount(l2_winners_labels))
            print(l2_winner)
            # k==1
            #l2_winner= np.argmin(l2_dis_sum)
            # Set result fir current prediction sample
            result[idx]= self.labels[l2_winner]

        # Return predicted labels
        return result





