
#%%
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

#%% [markdown]
# Read 1 image and display some statistics about it.

#%%
from matplotlib.image import imread
img = imread("./cats-dogs-dataset/test_set/cats/cat.4001.jpg")

data = {
    'description': 'Cats & Dogs dataset',
    'label': [ 'cat' ],
    'filename': [ '' ],
    'data': [ img ]
}

print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

#%% [markdown]
# Read in all images.

#%%
import glob
import ntpath
import re

data_path_train_cats = './cats-dogs-dataset/training_set/cats/cat.400*.jpg'
data_path_train_dogs = './cats-dogs-dataset/training_set/dogs/dog.400*.jpg'
data_path_train = './cats-dogs-dataset/training_set/**/*.400*.jpg'
data_path_test_cats = './cats-dogs-dataset/test_set/cats/cat.400*.jpg'
data_path_test_dogs = './cats-dogs-dataset/test_set/dogs/dog.400*.jpg'

X_train = []
y_train = []

for file in glob.glob(data_path_train):
    X_train.append(imread(file))
    
    label = re.search('^([^.]+)', ntpath.basename(file)).group(0)
    y_train.append(label)
# images = [imread(filepath) for file in glob.glob()]
print(X_train)

#%%


#%% [markdown]
# | Sidenote; data is already split, but maybe we should let sklearn split it using train_test_split, to prevent overfit/missed training data.

#%%
import matplotlib.pyplot as plt


#%%


#%% [markdown]
# Inspired by
# https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
# https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
# http://www.marekrei.com/blog/transforming-images-to-feature-vectors/

