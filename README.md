## Computer Vision Practice with Python
This repository helps me keep track of my progress with learning computer vision and python

---------------------------------------------------------------------------------------------------
### K Nearest Neighbours
The K nearest neighbours algorithm keeps a copy of all the training data supplied and their corresponding labels. It classifies a new data by comparing it to the available data and matching it to the data with the smallest distance (smallest error).

#### Implementation
An algorithm was implemented for K=1.
Training was carried out using CIFAR-10 dataset.
Testing was performed with 2 random images from Google and 2 images from the test batch of CIFAR-10.
The algorithm successfully identified one of the samples images from CIFAR-10 test batch, it failed on the other three.

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

---------------------------------------------------------------------------------------------------
