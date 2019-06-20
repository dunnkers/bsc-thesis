
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
from matplotlib.image import imread

# first 9 cats/dogs
data_path_train = './cats-dogs-dataset/training_set/**/*.?.jpg'
data_path_test = './cats-dogs-dataset/test_set/**/*.400*.jpg'

X_train = []
X_test = []
y_train = []
y_test = []

# load train data
for file in glob.glob(data_path_train):
    X_train.append(imread(file)) # read image

    # extract label from filename
    label = re.search('^([^.]+)', ntpath.basename(file)).group(0)
    y_train.append(label)

# load test data
for file in glob.glob(data_path_test):
    X_test.append(imread(file)) # read image

    # extract label from filename
    label = re.search('^([^.]+)', ntpath.basename(file)).group(0)
    y_test.append(label)

#%% [markdown]
# | Sidenote; data is already split, but maybe we should let sklearn split it using train_test_split, to prevent overfit/missed training data.

#%% [markdown]
# Plot the distribution of the training / test data.
# @from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

#%%
import matplotlib.pyplot as plt
def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5

    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]

    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'

    xtemp = np.arange(len(unique))

    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique)
    plt.xlabel('entity type')
    plt.ylabel(ylabel_text)

plt.suptitle('relative amount of photos per type')
plot_bar(y_train, loc='left')
plot_bar(y_test, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train)),
    'test ({0} photos)'.format(len(y_test))
])

#%% [markdown]
# Computing Histograms of Oriented Gradients (HOG's)

#%%
from skimage.feature import hog
from skimage.transform import rescale
import skimage

# map train data
img_show_idx = 0
im_original = X_train[img_show_idx]
X_train_hogged_images = []
def prepare(image):
    image = skimage.color.rgb2gray(image)
    image = rescale(image, 1/3, 0, 'reflect', True, False, False) # make smaller
    hogged, hogged_image = hog(
        image, pixels_per_cell=(12, 12),
        cells_per_block=(2,2),
        orientations=8,
        visualize=True,
        block_norm='L2-Hys')
    X_train_hogged_images.append(hogged_image)
    return hogged;

X_train = list(map(prepare, X_train))

# plotting...
im_hogged = X_train_hogged_images[img_show_idx]
hog = X_train[img_show_idx]
fig, ax = plt.subplots(1,2)
fig.set_size_inches(8,6)
# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for a in ax]
ax[0].imshow(im_original, cmap='gray')
ax[0].set_title('im')
ax[1].imshow(im_hogged, cmap='gray')
ax[1].set_title('hog_img')
plt.show()

print('number of pixels: ', im_original.shape[0] * im_original.shape[1])
print('number of hog features: ', hog.shape[0])

#%% [markdown]
# Inspired by
# https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
# https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
# http://www.marekrei.com/blog/transforming-images-to-feature-vectors/

