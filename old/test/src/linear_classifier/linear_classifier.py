
import numpy as np
import sys


from nodes.softmax import Softmax
from nodes.neg_natural_log import NegNatLog
from nodes.fc_layer import FCLayer
from nodes.multiclass_svm_loss import MulticlassSVMLoss
from utils.numeric_gradient import evaluate_gradient


class LinearClassifier:

    def __init__(self):
        pass

    def train(self, data, labels, *, learning_rate=1e-3, output_dim=10, no_of_epoch=1000000):
        """
            - data: ndarray(n,m). Train data matrix with each col a data sample
            - labels: ndarray(m,). Gtound truth labels for training data
        """
        # Data dim
        n= data.shape[0]
        m= data.shape[1]
        # Initialize weights
        W= np.random.randn(output_dim, n) / np.sqrt(m) * 0.01
        # print(str(W))
        # W= np.array([[0.01,-0.05,0.1,0.05],[0.7,0.2,0.05,0.16],[0,-0.45,-0.2,0.03]])
        # Initialize bias
        b= np.zeros(output_dim)
        # b= np.array([0,0.2,-0.3])
        # Augment W and b
        W_b= np.append(W, np.array([b]).T, axis=1)
        self.W_b= W_b

        softmax= Softmax()
        neg_nat_log= NegNatLog()
        fc_layer= FCLayer(W_b)
        multi_svm= MulticlassSVMLoss()
        
        ctr= 0
        aneal_epoch= 10
        print_epoch= 1

        for epoch in range(no_of_epoch):

            # # SOFTMAX
            # # Forward Prop
            # l_i= neg_nat_log.forward(softmax.forward(fc_layer.forward(data, W_b=self.W_b), labels))
            # loss= np.sum(l_i)/l_i.shape[0]
            # # Backprop
            # dW_b= fc_layer.backward(softmax.backward(neg_nat_log.backward(np.ones(m)/m)))

            # MULTICLASS SVM
            # Forward Prop
            l_i= multi_svm.forward(fc_layer.forward(data, W_b=self.W_b), labels)
            loss= np.sum(l_i)/l_i.shape[0]
            # Backprop
            dW_b= fc_layer.backward(multi_svm.backward(np.ones(m)/m))

            # # NUMERIC GRADIENT  
            # def func(W_b):
            #     # Loss per data sample
            #     # L_i= neg_nat_log.forward(softmax.forward(fc_layer.forward(data, W_b=W_b), labels))
            #     L_i= multi_svm.forward(fc_layer.forward(data, W_b=W_b), labels)
            #     # Total Loss
            #     return np.sum(L_i)/L_i.shape[0]
            # # Numeric Grad
            # dW_b_num= evaluate_gradient(func, self.W_b)
            # # Gadient error
            # dW_err= np.abs(dW_b-dW_b_num)/np.fmax(dW_b, dW_b_num)

            # Printing
            if epoch+1 >= print_epoch:
                # print(str(epoch))
                # print("max rel error: {0:3}".format(str(dW_err)))
                print("loss: {0:10}".format(str(loss)))
                print("learning rate: {0:10}".format(str(learning_rate)))
                print_epoch= print_epoch*2
                # Exit
                if loss <= 0.0001:
                    break

            # Update weights
            self.W_b += -learning_rate*dW_b
            # Aneal learning rate
            ctr+= 1
            if ctr >= aneal_epoch:
                learning_rate= learning_rate- learning_rate*0.01
                ctr= 0

            


    def predict(self, data):
        """
            - data: ndarray(n,m). Matrix with each col input to predict
        """
        pass
