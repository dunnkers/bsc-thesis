#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

# @FIXME or dynamically compute? analyze average image size?
SIZE_NORMAL_SHAPE = (1400, 700)

## Training
# Ground truth
gt = imread_collection('./data/groundtruth/image?.png', False)
# Supervised
sv = imread_collection('./data/supervised/image?.png', False)
# Unsupervised
usv = imread_collection('./data/unsupervised/output/output_image?.png', False)

print('gt size: {}, sv size: {}, usv size: {}'.format(
        gt.data.size, sv.data.size, usv.data.size
    ))

## Testing
# Ground truth
gt_test = imread_collection('./data/groundtruth/image1?.png', False)
# Supervised
sv_test = imread_collection('./data/supervised/image1?.png', False)
# Unsupervised
usv_test = imread_collection('./data/unsupervised/output/output_image1?.png', False)

print('gt_test size: {}, sv_test size: {}, usv_test size: {}'.format(
        gt_test.data.size, sv_test.data.size, usv_test.data.size
    ))

#%% [markdown]
# Plot an image of each data type.

#%%
from matplotlib import pyplot
from skimage.io import imshow

print('Raw image data for training sample at index = 0:')

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
# Transform image data: convert to grayscale, resize, rescale.

#%%
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu

class ResizeTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([resize(img, SIZE_NORMAL_SHAPE) for img in X])

class StretchTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([rescale_intensity(img) for img in X])

class ThresholdingTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([img > threshold_otsu(img) for img in X])

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [rgb2gray(img) for img in X]

resizer = ResizeTransform()
stretcher = StretchTransform()
grayify = RGB2GrayTransformer()
binarizer = LabelBinarizer()
thresholder = ThresholdingTransform()

## Training
# Transform Ground Truth
gt_grayed = grayify.fit_transform(gt)
gt_resized = resizer.fit_transform(gt_grayed)
gt_prepared = thresholder.fit_transform(gt_resized)
# Transform supervised
sv_resized = resizer.fit_transform(sv)
sv_prepared = thresholder.fit_transform(sv_resized)
# Transform unsupervised
usv_resized = resizer.fit_transform(usv)
usv_stretched = stretcher.fit_transform(usv_resized)
usv_prepared = thresholder.fit_transform(usv_stretched)

## Testing
# Transform Ground Truth
gt_transformed_test = grayify.transform(gt_test)
gt_resized_test = resizer.transform(gt_transformed_test)
# Transform supervised
sv_resized_test = resizer.transform(sv_test)
# Transform unsupervised
usv_resized_test = resizer.transform(usv_test)

#%% [markdown]
# Inspect new, resized and rescaled image data. Data is now in range
# of 0 to 1.

#%%
print('Prepared training image data for index = 0:')

pyplot.subplot(1, 3, 1).set_title("Ground truth")
imshow(gt_prepared[0])
print('Ground truth:\tshape {}\tmin,max({}, {})'.format(
        gt_prepared[0].shape, gt_prepared[0].min(), gt_prepared[0].max()
    ))

pyplot.subplot(1, 3, 2).set_title("Supervised")
imshow(sv_prepared[0])
print('Supervised:\tshape {}\tmin,max({}, {})'.format(
        sv_prepared[0].shape, sv_prepared[0].min(), sv_prepared[0].max()
    ))

pyplot.subplot(1, 3, 3).set_title("Unsupervised")
imshow(usv_prepared[0])
print('Unsupervised:\tshape {}\tmin,max({}, {})'.format(
        usv_prepared[0].shape, usv_prepared[0].min(), usv_prepared[0].max()
    ))

pyplot.show()


#%% [markdown]
# Combine two approaches into a feature-vector.

#%%
# @FIXME should probably be a sklearn transformation of sorts.
# -> sklearn.preprocessing.PolynomialFeatures perhaps
""" Combine 2D image pixel values, by addition. Return flat (feature) array. """
def combineImage(imgA, imgB):
    return np.array([
        a + b for a, b in zip(imgA.flatten(), imgB.flatten())
    ])

def combineImageSet(setA, setB):
    return np.array([
        combineImage(imgA, imgB) for imgA, imgB in zip(setA, setB)
    ])

def flattenImageSet(imgSet):
    return np.array([
        img.flatten() for img in imgSet
    ])

## Training
X_train = combineImageSet(sv_resized, usv_prepared)
y_train = flattenImageSet(gt_resized)

## Testing
X_test = combineImageSet(sv_resized_test, usv_resized_test)
y_test = flattenImageSet(gt_resized_test)

print('X_train shape: (per-pixel 2-feature vector)', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape: (per-pixel 2-feature vector)', X_test.shape)
print('y_test shape:', y_test.shape)

#%% [markdown]
# Train a classifier and predict.

#%%
from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
# sgd_clf.fit(X_train, y_train)
# y_pred = sgd_clf.predict(X_test)

from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

clf = SVC(gamma='auto')
sgd_regr = SGDRegressor()
multout_clf = MultiOutputClassifier(sgd_regr)
multout_clf.fit(X_train, y_train)
y_pred = multout_clf.predict(X_test)
y_pred

print('End of program stub.')