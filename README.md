## Computer Vision Practice with Python
This repository helps me keep track of my progress with learning computer vision and python

---------------------------------------------------------------------------------------------------
### K Nearest Neighbours
The K nearest neighbours algorithm keeps a copy of all the training data supplied and their corresponding labels. It classifies a new data by comparing it to the available data and matching it to the data with the smallest distance (smallest error).

#### Implementation
An alogrithm was implemented using both l1 & l2 distances (can switch alternatively) and variable 'k'. 
Training was carried out using 49,000 images from the CIFAR-10 dataset, while hyperparameters were tuned with the remaining 1,000.
Testing was performed with 6 random images from Google and the entire dataset from the test batch of CIFAR-10.

##### Related scripts
directory: /nearest_neighbour
- nearest_neighbour.py
- hyper_params.py
- run.py

   
#### Python learnings
* __Import class from local python file in the same folder__: To do this, a `__init__.py` file has to be created in the directory to enable imports from this directory. To import from sub-directories, each sub-directory must also have it's own `__init__.py` file.
```python
# Root directory
from file_name import ClassName
# Sub-directory
from directory_nsme.file_name import ClssName
```

* __Get list of keys from a Dictionary__: The `list()` function returns the list of keys available in a dictionary if called with the dictionary as an argument.
```python
dict= {'name': 'dictionary, 'function': 'nada!'}
keys_list= list(dict)
```

* __Get index of numpy array with smallest element__: The `argmin()` method of numpy returns the index of the smallest element. `argmax()` does the reverse.

* __Get index of sorted numpy array__: The `argsort()` method returns an index of sorted array elements (does not actually sort the array).

* __Get frequency of occurence of numpy array elements__: The `binsort()` method returns the frequency of occurence of array elements as a numpy array with the element as the index and it's frequency as the value for that index.

* __Get maximum element from numpy array__: The `amax()` method returns the maximum element from a numpy array.


```
min_index= np.argmin(np_arr)
sorted_index= np.argsort(np_arr)
most_frequent_3= np.binsort(np_arr)[:3]
```



---------------------------------------------------------------------------------------------------
### Linear Classifier
A linear classifier algorithm uses a linear function to classify data by assigning 'scores' to each class, the correct class is expected to have the highest score. The weighting matrix is trained on the training data by adjusting its parameters such that the lowest loss is obtained.  
   

A loss function is a quantitative evaluation of how well a neural network classifies training data. Common loss functions include the multi-class SVM classifier and the Softmax classifier. Since it is possible for diffrent weighting matrices to accrue the same loss, a regularization term is often included in the loss function. This term help to expresses preference of some W over another. A common chioce is the L_2 norm of W.    
**Multi-class SVM Loss**: Wants the score of the correct class to be higher than the other classes by some pre-defined margin. The loss for a data sample of index *i* where *s* is the scores vector and *y_i* is the index of the correct class is given as:   
![equation](https://latex.codecogs.com/gif.latex?L_%7Bi%7D%3D%20%5Csum_%7Bj%5Cneq%20y_%7Bi%7D%7D%5E%7B%20%7Dmax%280%2C%20s_%7Bj%7D%20-%20s_%7By_%7Bi%7D%7D%20&plus;%20%5CDelta%20%29)  

**Softmax-based Cross Entropy Loss**: This kind of loss function interpretes the class scores as unnormalized probabilities for each class. It computes the loss of a data sample of index *i* where *s* is the scores vector and *y_i* is the index of the ground truth as shown below:   
![equation](https://latex.codecogs.com/gif.latex?L_%7Bi%7D%3D%20-log%5Cleft%28%5Cfrac%7Be%5E%7Bs_%7By_%7Bi%7D%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%20%7De%5E%7Bs_%7Bj%7D%7D%7D%5Cright%29)  
As evident in the equation above, the higher the score of the ground truth is as compared to the other classes, the lower the loss and vice versa.  

  
The total loss of a batch of training data factoring in the regularization loss *R(W)* can then be computed as:  
![equation](https://latex.codecogs.com/gif.latex?L%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5E%7BN%7DL_%7Bi%7D%20&plus;%20R%5Cleft%28W%20%5Cright%20%29)  

#### Implementation  