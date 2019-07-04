#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

#%%
from skimage.io import imread_collection

# @FIXME or dynamically compute? analyze average image size?
SIZE_NORMAL_SHAPE = (1400, 700)
RESCALE_FACTOR = 1.0 / 7.0
# 1.0 / 4.0 = (350,175) => 61250 elements
# 1.0 / 7.0 = (200,100) => 20000 elements
SAMPLES_AMOUNT = 100

print('Resized shape: {}'.format(SIZE_NORMAL_SHAPE))
print('Rescale factor: {:.2f}'.format(RESCALE_FACTOR))
print('Sample amount: {:.2f}'.format(SAMPLES_AMOUNT))

## Training
# Ground truth
gt = imread_collection('./data/groundtruth/image1?.png', False)
# Supervised
sv = imread_collection('./data/supervised/image1?.png', False)
# Unsupervised
usv = imread_collection('./data/unsupervised/output/output_image1?.png', False)

print('gt size: {}, sv size: {}, usv size: {}'.format(
        gt.data.size, sv.data.size, usv.data.size
    ))
assert(gt.data.size == sv.data.size == usv.data.size)

## Testing
# Ground truth
gt_test = imread_collection('./data/groundtruth/image?.png', False)
# Supervised
sv_test = imread_collection('./data/supervised/image?.png', False)
# Unsupervised
usv_test = imread_collection('./data/unsupervised/output/output_image?.png', False)

print('gt_test size: {}, sv_test size: {}, usv_test size: {}'.format(
        gt_test.data.size, sv_test.data.size, usv_test.data.size
    ))
assert(gt_test.data.size == sv_test.data.size == usv_test.data.size)

#%% [markdown]
# Plot an image of each data type.

#%%
from matplotlib import pyplot
from skimage.io import imshow

def plotImgColumn(title, img, idx, cols=3, hist=True):
    # print('{}:\tshape={}\tminmax=({}, {})'.format(
    #         title, img.shape, img.min(), img.max()))
    rows = 2 if hist else 1
    pyplot.subplot(rows, cols, idx).set_title(title)
    imshow(img)
    if hist:    
        ax_hist = pyplot.subplot(rows, cols, cols + idx, label=title)
        ax_hist.hist(img.ravel(), bins=128)

# print('Raw image data for training sample at index = 0:')
# plotImgColumn("Ground truth", gt[0], 1)
# plotImgColumn("Supervised", sv[0], 2)
# plotImgColumn("Unsupervised", usv[0], 3)
# pyplot.show()

#%% [markdown]
# Transform image data: convert to grayscale, resize, rescale, threshold.
# See [https://en.wikipedia.org/wiki/Otsu%27s_method](https://en.wikipedia.org/wiki/Otsu%27s_method).

#%%
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import StratifiedShuffleSplit    
from sklearn.utils import resample

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

class FlattenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [img.flatten() for img in X]

class ZipperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        a, b = X
        return np.array((a.flatten(), b.flatten())).T

class SamplerTransformer(BaseEstimator, TransformerMixin):
    """ Sample size set to the class minority by default.
        Can specify custom sample size. """
    def __init__(self, sample_size=None):
        self.sample_size = sample_size

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        # Resamples classes to match the size of the smallest class,
        # aka the minority.
        def sample(vector):
            classes, counts = np.unique(vector, return_counts=True)
            minority = np.argmin(counts)

            # resample with stratify is sklearn >= v0.21.2
            # resampled = resample(vector, ...)
            splitted = [
                np.where(vector == classname)[0] for classname in classes
            ]

            sample_size = self.sample_size or counts[minority]

            resampled = np.array([
                resample(split, n_samples=sample_size, random_state=41)
                    for split in splitted
            ])

            return resampled.flatten()

        return np.array([sample(vector) for vector in X])

# Select certain indices in dataset
class SelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        data, indexes = X
        return np.array([
            vector[select] for vector, select in zip(data, indexes)
            ])

grayify = RGB2GrayTransformer()
resizer = ResizeTransform()
rescaler = RescalerTranform()
thresholder = ThresholdingTransform() # use Binarize or sim; perf
vectorize = ToVectorTransformer()
zipper = ZipperTransformer()
flatten = FlattenTransformer()
sampler = SamplerTransformer(sample_size=SAMPLES_AMOUNT)
selector = SelectTransformer()

##### Training
### Transform Ground Truth images
gt_grayed = grayify.fit_transform(gt)
gt_resized = resizer.fit_transform(gt_grayed)
gt_rescaled = rescaler.fit_transform(gt_resized)
gt_prepared = thresholder.fit_transform(gt_rescaled)
# Sample
gt_flat_img = flatten.fit_transform(gt_prepared)
sample_idxs = sampler.fit_transform(gt_flat_img)

### Transform SV and USV images
sv_resized = resizer.fit_transform(sv)
sv_rescaled = rescaler.fit_transform(sv_resized)
usv_resized = resizer.fit_transform(usv)
usv_rescaled = rescaler.fit_transform(usv_resized)
# Sample
sv_flat_img = flatten.fit_transform(sv_rescaled)
usv_flat_img = flatten.fit_transform(usv_rescaled)

### Select indices per-image
gt_selected = selector.fit_transform((gt_flat_img, sample_idxs))
sv_selected = selector.fit_transform((sv_flat_img, sample_idxs))
usv_selected = selector.fit_transform((usv_flat_img, sample_idxs))

### Feature vectors
X_train_all = zipper.fit_transform((sv_selected, usv_selected))
y_train_all = vectorize.fit_transform(gt_selected)

##### Testing
### Transform Ground Truth images
gt_test_grayed = grayify.transform(gt_test)
gt_test_resized = resizer.transform(gt_test_grayed)
gt_test_rescaled = rescaler.transform(gt_test_resized)
gt_test_prepared = thresholder.transform(gt_test_rescaled)

### Transform SV and USV images
sv_test_resized = resizer.transform(sv_test)
sv_test_rescaled = rescaler.transform(sv_test_resized)
usv_test_resized = resizer.transform(usv_test)
usv_test_rescaled = rescaler.transform(usv_test_resized)

### Feature vectors
X_test_all = zipper.transform((sv_test_rescaled, usv_test_rescaled))
y_test_all = vectorize.transform(gt_test_prepared)

X_train = X_train_all
y_train = y_train_all
X_test = X_test_all
y_test = y_test_all

print('X_train:\t{}\tsize {}'.format(X_train.shape, X_train.size))
print('y_train:\t{}\tsize {}'.format(y_train.shape, y_train.size))
print('X_test:\t\t{}\tsize {}'.format(X_test.shape, X_test.size))
print('y_test:\t\t{}\tsize {}'.format(y_test.shape, y_test.size))

# Picking random samples

#%% [markdown]
# Train a classifier and predict.

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# train support-vector-machine
svm = SVC(gamma = 'auto')
svm.fit(X_train, y_train)

#%% [markdown]
# Predict.

#%% 
predictions = svm.predict(X_test)

# accuracy score
acc_score = accuracy_score(y_test, predictions)
print(acc_score)

#%%
from os.path import splitext, basename

def reconstructImages(vectors):
    original_shape = (
            np.array(SIZE_NORMAL_SHAPE) * RESCALE_FACTOR
        ).astype('int')
    images = [np.reshape(vector, original_shape) for vector in vectors]
    return images

vectors = np.split(predictions, gt_test.data.size)
truth_vals = np.split(y_test,   gt_test.data.size)
reconstructed_images = reconstructImages(vectors)

for idx, im in enumerate(reconstructed_images):
    img_acc = accuracy_score(truth_vals[idx], vectors[idx])
    plotImgColumn("Ground truth", gt_test[idx], 1, hist=False, cols=4)
    plotImgColumn("Supervised", sv_test[idx], 2, hist=False, cols=4)
    plotImgColumn("Unsupervised", usv_test[idx], 3, hist=False, cols=4)
    plotImgColumn("Prediction", im, 4, hist=False, cols=4)

    # scale_factor=1.0/4.0; (-1000, 450)
    # scale_factor=1.0/7.0; (-500, 250)
    pyplot.text(-500, 250, 'X_train={}, accuracy={:.4f}%'.format(
        X_train.shape, img_acc*100
    ))
    file, ext = splitext(gt.files[idx])
    pyplot.savefig('/home/s2995697/Downloads/{}_fused{}'.format(
        basename(file), ext
    ))
    pyplot.show()
