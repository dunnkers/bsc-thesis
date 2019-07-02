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

print('Raw image data for training sample at index = 3:')
plotImgColumn("Ground truth", gt[3], 1)
plotImgColumn("Supervised", sv[3], 2)
plotImgColumn("Unsupervised", usv[3], 3)
pyplot.show()

print('Raw image data for training sample at index = 4:')
plotImgColumn("Ground truth", gt[4], 1)
plotImgColumn("Supervised", sv[4], 2)
plotImgColumn("Unsupervised", usv[4], 3)
pyplot.show()

print('Raw image data for training sample at index = 8:')
plotImgColumn("Ground truth", gt[8], 1)
plotImgColumn("Supervised", sv[8], 2)
plotImgColumn("Unsupervised", usv[8], 3)
pyplot.show()

#%% [markdown]
# Transform image data: convert to grayscale, resize, rescale, threshold.
# See [https://en.wikipedia.org/wiki/Otsu%27s_method](https://en.wikipedia.org/wiki/Otsu%27s_method).

#%%
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_triangle

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
        return np.array([img > threshold_triangle(img) for img in X])

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [rgb2gray(img) for img in X]

class FlattenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return np.array([img.flatten() for img in X])

class CombineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        setA, setB = X
        return np.array([
            np.logical_and(imgA, imgB) for imgA, imgB in zip(setA, setB)
        ])

grayify = RGB2GrayTransformer()
resizer = ResizeTransform()
stretcher = StretchTransform()
thresholder = ThresholdingTransform()
flattener = FlattenTransformer()
combiner = CombineTransformer()

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
## Convert to feature vectors (flatten)
y_train = flattener.fit_transform(gt_prepared)
print('y_train:\t{}'.format(y_train.shape))
to_combine = (sv_prepared, usv_prepared)
X_train_combined = combiner.fit_transform(to_combine)
X_train = flattener.fit_transform(X_train_combined)
print('X_train:\t{}'.format(X_train.shape))

## Testing
# Transform Ground Truth
gt_transformed_test = grayify.transform(gt_test)
gt_resized_test = resizer.transform(gt_transformed_test)
# Transform supervised
sv_resized_test = resizer.transform(sv_test)
# Transform unsupervised
usv_resized_test = resizer.transform(usv_test)

print('Prepared image data for training sample at index = 0:')
plotImgColumn("Ground truth", gt_prepared[0], 1, hist=False, cols=4)
plotImgColumn("Supervised", sv_prepared[0], 2, hist=False, cols=4)
plotImgColumn("Unsupervised", usv_prepared[0], 3, hist=False, cols=4)
plotImgColumn("Combined", X_train_combined[0], 4, hist=False, cols=4)
pyplot.show()

print('Prepared image data for training sample at index = 3:')
plotImgColumn("Ground truth", gt_prepared[3], 1, hist=False, cols=4)
plotImgColumn("Supervised", sv_prepared[3], 2, hist=False, cols=4)
plotImgColumn("Unsupervised", usv_prepared[3], 3, hist=False, cols=4)
plotImgColumn("Combined", X_train_combined[3], 4, hist=False, cols=4)
pyplot.show()

print('Prepared image data for training sample at index = 4:')
plotImgColumn("Ground truth", gt_prepared[4], 1, hist=False, cols=4)
plotImgColumn("Supervised", sv_prepared[4], 2, hist=False, cols=4)
plotImgColumn("Unsupervised", usv_prepared[4], 3, hist=False, cols=4)
plotImgColumn("Combined", X_train_combined[4], 4, hist=False, cols=4)
pyplot.show()

print('Prepared image data for training sample at index = 8:')
plotImgColumn("Ground truth", gt_prepared[8], 1, hist=False, cols=4)
plotImgColumn("Supervised", sv_prepared[8], 2, hist=False, cols=4)
plotImgColumn("Unsupervised", usv_prepared[8], 3, hist=False, cols=4)
plotImgColumn("Combined", X_train_combined[8], 4, hist=False, cols=4)
pyplot.show()

#%% [markdown]
# Analyze current performance of both approaches.

#%%
def accuracy(truth_vector, test_vector):
    return np.sum(test_vector == truth_vector) / len(truth_vector) * 100

def accuracySet(setA, setB):
    return np.array([
        accuracy(
            imgA.flatten(), imgB.flatten()
        ) for imgA, imgB in zip(setA, setB)
    ])
# use this: from sklearn.metrics import accuracy_score

acc_sv = accuracySet(gt_prepared, sv_prepared)
acc_usv = accuracySet(gt_prepared, usv_prepared)

print('Supervised accuracy:')
print('\tmean={0:.2f}%\tminmax=({1:.2f}, {2:.2f})\tvariance={3:.2f}'.format(
    acc_sv.mean(), acc_sv.min(), acc_sv.max(), acc_sv.var()
))
print('Unsupervised accuracy:')
print('\tmean={0:.2f}%\tminmax=({1:.2f}, {2:.2f})\tvariance={3:.2f}'.format(
    acc_usv.mean(), acc_usv.min(), acc_usv.max(), acc_usv.var()
))

#%% [markdown]
# Train a classifier and predict.

#%%
# from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
# sgd_clf.fit(X_train, y_train)
# y_pred = sgd_clf.predict(X_test)

# from sklearn.linear_model import SGDRegressor
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.svm import SVC

# clf = SVC(gamma='auto')
# sgd_regr = SGDRegressor()
# multout_clf = MultiOutputClassifier(sgd_regr)
# multout_clf.fit(X_train, y_train)
# # y_pred = multout_clf.predict(X_test)
# # y_pred

# print('End of program stub.')