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

* __Get index of numpy array with smallest element__: The `argmin()` method of numpy returns the index of the smallest element.

* __Get index of sorted numpy array__: The `argsort()` method returns an index of sorted array elements (does not actually sort the array).

* __Get frequency of occurence of numpy array elements__: The `binsort()` method returns the frequency of occurence of array elements as a numpy array with the element as the index and it's frequency as the value for that index.

*__Get maximum element from numpy array__: The `amax()` method returns the maximum element from a numpy array.


```
min_index= np.argmin(np_arr)
sorted_index= np.argsort(np_arr)
most_frequent_3= np.binsort(np_arr)[:3]
```



---------------------------------------------------------------------------------------------------
