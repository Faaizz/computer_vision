
# THIS SCRIPT SUPPLIES TRAINING, AND TESTING DATA FROM CIFAR-10 DATASET
# THE DATASET IS LOCATED IN "/cifar-10-batches-py" FOLDER IN THE ROT DIRECTORY OF THE PROJECT

import numpy as np
import os
import sys

# Get current path
root_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root path
root_dir= root_dir.replace("/src/data", "")

def unpickle(file):
    """
        Unpickles datafile
    """
    import pickle
    with open(file, 'rb') as f:
        data_dict= pickle.load(f, encoding="bytes")
    return data_dict


def get_training_data(*, no_of_batches=1):
    """
        Returns training dataset(s) as dictionary of numpy-array values in which column of 'data' represents a single data sample
        no_of_batches= number of dataset batches to return
    """

    if no_of_batches > 5:
        raise Exception("CIFAR-10 has only 5 dataset batches. Please specify a number not greater than 5.")

    # First dataset batch
    dataset= unpickle(root_dir + '/cifar-10-batches-py/data_batch_1')
    key_list= list(dataset)
    data_key= key_list[2]
    labels_key= key_list[1]
    data= np.array(dataset[data_key])
    labels= np.array(dataset[labels_key])

    # More dataset batches- if required
    for idx in range(2, no_of_batches+1):
        dataset= unpickle(root_dir + '/cifar-10-batches-py/data_batch_' + str(idx))
        data= np.append(data,dataset[data_key],axis=0)
        labels= np.append(labels,dataset[labels_key],axis=0)

    return {"data":data.T,"labels":labels}


def get_test_data():
    """
        Returns testing data as a dictionary of numpy-array values in which column of 'data' represents a single data sample
    """
    test_dataset= unpickle(root_dir + '/cifar-10-batches-py/test_batch')
    test_key_list= list(test_dataset)

    return {"data":np.array(test_dataset[test_key_list[2]]).T, "labels":np.array(test_dataset[test_key_list[1]])}

if __name__ == "__main__":
    print("Root dir: {0}".format(root_dir))


def get_dir():
    print("Root dir: {0}".format(root_dir))