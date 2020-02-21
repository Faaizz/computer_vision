import numpy as np

class NearestNeighbour:

    def __init__(self):
        pass

    def train(self, data, labels):
        """ data is N x D while labels is 1-dim with length N """
        # Simply remember the training data
        self.data= data
        slef.labels= labels

    def predict(self, data):
        """ data is N x D where N is the number of samples for which to predict """
        # Setup result array
        result= np.array(data.shape[0], dtype=self.labels.dtype)
        # Loop through the data to predict
        for idx in range(data.shape[0]):
            # Find L2 distance between each col of training data and current prediction sample (broadcasting)
            l2_dis= np.sqrt(np.square(self.data-data[idx]))
            # Sum up the col distances on each row
            l2_dis_sum= np.sum(l2_dis, axis=1)
            # Identify the row with the smallest distance
            l2_winner= np.argmin(l2_dis_sum)
            # Set result fir current prediction sample
            result[idx]= self.labels[idx]

        # Return predicted labels
        return result





