#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

# @FIXME or dynamically compute? analyze average image size?
SIZE_NORMAL_SHAPE = (1400, 700)
RESCALE_FACTOR = 1.0 / 4.0
SAMPLES_AMOUNT = 1000
# TRAIN_SAMPLES = 7
# TEST_SAMPLES = 2

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
from sklearn.model_selection import StratifiedShuffleSplit    

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
# Transform Ground Truth images
gt_grayed = grayify.fit_transform(gt)
gt_resized = resizer.fit_transform(gt_grayed)
gt_rescaled = rescaler.fit_transform(gt_resized)
gt_prepared = thresholder.fit_transform(gt_rescaled)
## Truth vector
y_train_all = vectorize.fit_transform(gt_prepared)
# Transform sv and usv into 1
# Transform SV and USV images
sv_resized = resizer.fit_transform(sv)
sv_rescaled = rescaler.fit_transform(sv_resized)
usv_resized = resizer.fit_transform(usv)
usv_rescaled = rescaler.fit_transform(usv_resized)
## Feature vector
X_train_all = zipper.fit_transform((sv_rescaled, usv_rescaled))

# SAMPLING
# @FIXME split first, then transform. DONT fit on test data.

# def stratified_split(y, train_ratio):

#     def split_class(y, label, train_ratio):
#         indices = np.flatnonzero(y == label)
#         n_train = int(indices.size*train_ratio)
#         train_index = indices[:n_train]
#         test_index = indices[n_train:]
#         return (train_index, test_index)

#     idx = [split_class(y, label, train_ratio) for label in np.unique(y)]
#     train_index = np.concatenate([train for train, _ in idx])
#     test_index = np.concatenate([test for _, test in idx])
#     return train_index, test_index

# X = X_train_all
# y = y_train_all
# prepared_shape = np.array(SIZE_NORMAL_SHAPE) * RESCALE_FACTOR
# prepared_length = np.prod(prepared_shape)
# train_samples = 10000
# test_samples = 1000
# train_ratio = float(train_samples)/(train_samples + test_samples)
# train_index, test_index = stratified_split(y, train_ratio)
# y_train = y[train_index]
# y_test = y[test_index]

sss = StratifiedShuffleSplit(train_size=10000, n_splits=1, 
                             test_size=1000, random_state=0)  
X = X_train_all
y = y_train_all
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

from scipy import stats
print(stats.itemfreq(y_train))
print(stats.itemfreq(y_test))
print('end')

# Picking random samples

# indices = sample_without_replacement(prepared_length, SAMPLES_AMOUNT)
# samples = np.take(X_train_all, indices, axis=0)
# print(samples.shape)

#%% [markdown]
# Train a classifier and predict.

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# train support-vector-machine
svm = SVC(gamma = 'auto')
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

# # accuracy score
acc_score = accuracy_score(y_test, predictions)
print(acc_score)
acc_score
