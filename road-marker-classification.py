#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

SIZE_NORMAL_SHAPE = (1400, 700) # .. or dynamically compute

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

print('Raw image data for index = 0:')

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
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.color import rgb2gray
from skimage.transform import resize

class ResizeTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([resize(img, SIZE_NORMAL_SHAPE) for img in X])

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [rgb2gray(img) for img in X]

resizer = ResizeTransform()
grayify = RGB2GrayTransformer()

# Transform Ground Truth
gt_transformed = grayify.fit_transform(gt)
gt_resized = resizer.fit_transform(gt_transformed)
# Transform supervised
sv_resized = resizer.fit_transform(sv)
# Transform unsupervised
usv_resized = resizer.fit_transform(usv)


#%% [markdown]
# Inspect new, resized and rescaled image data. Data is now in range
# of 0 to 1.

#%%
print('Prepared image data for index = 0:')

print('Ground truth:\tshape {}'.format(
        gt_resized[0].shape
    ))
pyplot.subplot(2, 3, 1).set_title("Ground truth")
imshow(gt_resized[0])
ax_hist = pyplot.subplot(2, 3, 4)
ax_hist.hist(gt_resized[0].ravel(), bins=32)

print('Supervised:\tshape {}'.format(
        sv_resized[0].shape
    ))
pyplot.subplot(2, 3, 2).set_title("Supervised")
imshow(sv_resized[0])
ax_hist = pyplot.subplot(2, 3, 5)
ax_hist.hist(sv_resized[0].ravel(), bins=256)

print('Unsupervised:\tshape {}'.format(
        usv_resized[0].shape
    ))
pyplot.subplot(2, 3, 3).set_title("Unsupervised")
imshow(usv_resized[0])
ax_hist = pyplot.subplot(2, 3, 6)
ax_hist.hist(usv_resized[0].ravel(), bins=32)

pyplot.show()


#%% [markdown]
# Combine two approaches into a feature-vector.

#%%
# @FIXME should probably be a sklearn transformation of sorts.
def combine(sv_img, usv_img):
    return np.array([
        (sv_pixel, usv_pixel)
        for sv_pixel, usv_pixel in zip(sv_img.flatten(), usv_img.flatten())
    ])

X_train = np.array([
    combine(sv_img, usv_img)
        for sv_img, usv_img in zip(sv_resized, usv_resized)
    ])

print('X_train shape: (per-pixel 2-feature vector)')
print(X_train.shape)

#%%
print('End of program stub.')