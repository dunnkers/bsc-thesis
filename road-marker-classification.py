#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

# Ground truth
gt = imread_collection('./data/groundtruth/image?.png', False)
# Supervised
sv = imread_collection('./data/supervised/image?.png', False)
# Unsupervised
usv = imread_collection('./data/unsupervised/output/output_image?.png', False)

print('gt size: {}, sv size: {}, usv size: {}'.format(
        gt.data.size, sv.data.size, usv.data.size
    ))

#%% [markdown]
# Plot an image of each data type.

#%%
from matplotlib import pyplot
from skimage.io import imshow

print('Raw image data:')

pyplot.subplot(2, 3, 1).set_title("Ground truth")
imshow(gt[0])
ax_hist = pyplot.subplot(2, 3, 4)
ax_hist.hist(gt[0].ravel(), bins=256)
print('Ground truth:\tshape {}\tmin,max({}, {})'.format(
        gt[0].shape, gt[0].min(), gt[0].max()
    ))

pyplot.subplot(2, 3, 2).set_title("Supervised")
imshow(sv[0])
ax_hist = pyplot.subplot(2, 3, 5)
ax_hist.hist(sv[0].ravel(), bins=256)
print('Supervised:\tshape {}\tmin,max({}, {})'.format(
        sv[0].shape, sv[0].min(), sv[0].max()
    ))

pyplot.subplot(2, 3, 3).set_title("Unsupervised")
imshow(usv[0])
ax_hist = pyplot.subplot(2, 3, 6)
ax_hist.hist(usv[0].ravel(), bins=256)
print('Unsupervised:\tshape {}\tmin,max({}, {})'.format(
        usv[0].shape, usv[0].min(), usv[0].max()
    ))

pyplot.show()

#%% [markdown]
# Transform image data into feature vectors.

#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.color import rgb2gray

def gray(image, visualize=False):
    grayscale_image = rgb2gray(image)
    feature_vector = grayscale_image.flatten()

    if visualize:
        return feature_vector, grayscale_image
    else:
        return feature_vector

def unitize(image, visualize=False):
    feature_vector = image.flatten()

    if visualize:
        return feature_vector, image
    else:
        return feature_vector

def reconstruct_image(feature_vector, shape):
    return np.reshape(feature_vector, shape)

class FlattenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [img.ravel() for img in X]

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [gray(img) for img in X]

grayify = RGB2GrayTransformer()
gt_transformed = grayify.fit_transform(gt)

_, gt_im = gray(gt[0], visualize=True)
print('Ground truth:\tshape {}'.format(
        gt_im.shape
    ))
pyplot.subplot(2, 2, 1).set_title("Ground truth")
imshow(gt_im)
ax_hist = pyplot.subplot(2, 2, 3)
ax_hist.hist(gt_im.ravel(), bins=256)

# # (!) needs a feature vector to work..
# print(scaler.fit(sv))
# print(scaler.data_max_)
# print(scaler.transform(sv))
# normalize = Normalizer()
# sv_transformed = normalize.fit_transform(sv)

scaler = MinMaxScaler()
standardizer = StandardScaler()
flatten = FlattenTransformer()
sv_flattened = flatten.fit_transform(sv)
sv_normalized = scaler.fit_transform([sv_flattened[0]])
_, sv_im = unitize(sv[0], visualize=True)
print('Ground truth:\tshape {}'.format(
        sv_im.shape
    ))
pyplot.subplot(2, 2, 2).set_title("Supervised")
imshow(sv_im)
ax_hist = pyplot.subplot(2, 2, 4)
ax_hist.hist(sv_im.ravel(), bins=256)

print('End of program stub.')

#%%
