#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

# @FIXME or dynamically compute? analyze average image size?
SIZE_NORMAL_SHAPE = (1400, 700)
RESCALE_FACTOR = 1.0 / 4.0
SAMPLES_AMOUNT = 1000

## Training
# Ground truth
gt = imread_collection('./data/groundtruth/image1.png', False)
# Supervised
sv = imread_collection('./data/supervised/image1.png', False)
# Unsupervised
usv = imread_collection('./data/unsupervised/output/output_image1.png', False)

print('gt size: {}, sv size: {}, usv size: {}'.format(
        gt.data.size, sv.data.size, usv.data.size
    ))

## Testing
# Ground truth
gt_test = imread_collection('./data/groundtruth/image10.png', False)
# Supervised
sv_test = imread_collection('./data/supervised/image10.png', False)
# Unsupervised
usv_test = imread_collection('./data/unsupervised/output/output_image10.png', False)

print('gt_test size: {}, sv_test size: {}, usv_test size: {}'.format(
        gt_test.data.size, sv_test.data.size, usv_test.data.size
    ))

#%% [markdown]
# Plot an image of each data type.

#%%
from matplotlib import pyplot
from skimage.io import imshow

def plotImgColumn(title, img, idx, cols=3, hist=True):
    print('{}:\tshape={}\tminmax=({}, {})'.format(
            title, img.shape, img.min(), img.max()))
    rows = 2 if hist else 1
    pyplot.subplot(rows, cols, idx).set_title(title)
    imshow(img)
    if hist:    
        ax_hist = pyplot.subplot(rows, cols, cols + idx, label=title)
        ax_hist.hist(img.ravel(), bins=128)

print('Raw image data for training sample at index = 0:')
plotImgColumn("Ground truth", gt[0], 1)
plotImgColumn("Supervised", sv[0], 2)
plotImgColumn("Unsupervised", usv[0], 3)
pyplot.show()

#%% [markdown]
# Transform image data: convert to grayscale, resize, rescale, threshold.
# See [https://en.wikipedia.org/wiki/Otsu%27s_method](https://en.wikipedia.org/wiki/Otsu%27s_method).

#%%
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import sample_without_replacement

from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from skimage.filters import threshold_yen

class ResizeTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([resize(img, SIZE_NORMAL_SHAPE) for img in X])

class RescalerTranform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([rescale(img, RESCALE_FACTOR) for img in X])

class ThresholdingTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([img > threshold_yen(img) for img in X])

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [rgb2gray(img) for img in X]

class ToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X.flatten()

class ZipperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        a, b = X
        return np.array((a.flatten(), b.flatten())).T

grayify = RGB2GrayTransformer()
resizer = ResizeTransform()
rescaler = RescalerTranform()
thresholder = ThresholdingTransform()
vectorize = ToVectorTransformer()
zipper = ZipperTransformer()

## Training
# Transform Ground Truth
gt_grayed = grayify.fit_transform(gt)
gt_resized = resizer.fit_transform(gt_grayed)
gt_rescaled = rescaler.fit_transform(gt_resized)
gt_prepared = thresholder.fit_transform(gt_rescaled)
## Create feature vector
y_train_all = vectorize.fit_transform(gt_prepared)
# Transform sv and usv into 1
sv_resized = resizer.fit_transform(sv)
sv_rescaled = rescaler.fit_transform(sv_resized)
usv_resized = resizer.fit_transform(usv)
usv_rescaled = rescaler.fit_transform(usv_resized)
X_train_all = zipper.fit_transform((sv_rescaled, usv_rescaled))

# Picking random samples
prepared_shape = np.array(SIZE_NORMAL_SHAPE) * RESCALE_FACTOR
prepared_length = np.prod(prepared_shape)

indices = sample_without_replacement(prepared_length, SAMPLES_AMOUNT)
samples = np.take(X_train_all, indices, axis=0)
print(samples.shape)

#%% [markdown]
# Train a classifier and predict.

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# train support-vector-machine
svm = SVC(gamma = 'auto')
svm.fit(X_train, y_train)
# predictions = svm.predict(X_test)

# # accuracy score
# accuracy_score(Y_validation, predictions)
