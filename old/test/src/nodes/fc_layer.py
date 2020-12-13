
# THIS SCRIPT PROVIDES A FULLY CONNECTED LAYER IN A NEURAL NETWORK

import numpy as np

class FCLayer:

    def __init__(self, weights_biases, *, activ_func=lambda x: x):
        """
            Initialize Layer
            -weights_biases: ndarray(no_of_neurons, input_dim). Augmented weghts & biases matrix
            -activ_func: func. activation function
        """
        self.W_b= weights_biases
        self.activ_func= activ_func

    def forward(self, data):
        """
            Forward propagation
            -data: ndarray(input_dim, no_of_inputs)
            -W_b: allow augmented weights-biases matrix to be specified. Primarily for gradient checking

            Return: ndarray(no_of_neurons, no_of_inputs)
        """
        # If W_b is not specified, use the one provided on initialization
        # if W_b.all() == 0:
        #     W_b= self.W_b
        # else:
            # self.W_b= W_b
        W_b= self.W_b

        # Extend data to accomodate augmented weight
        data= np.append(data, np.array([np.ones(data.shape[1])]), axis=0)
        # Store data for backprop
        self.data= data

        pre_activ= np.dot(W_b,data)
        # activate
        return self.activ_func(pre_activ)

    def backward(self, up_grad):
        """
            Compute local gradient
            -up_grad: ndarray(no_of_neurons,no_of_inputs). Upstream gradient
        """
        return np.dot(up_grad, self.data.T)
        # up_grad= np.sum(up_grad, axis=1)
        # up_grad= up_grad[:,0]
        # return (up_grad*loc_grad.T).T




    