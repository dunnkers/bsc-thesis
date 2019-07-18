#%%
import constants as const
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.io import imread_collection
from skimage.util import img_as_bool
from sklearn.utils import resample
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from tqdm.auto import tqdm
import pickle
from sklearn.model_selection import KFold
from os.path import join, basename
from re import findall
from constants import CACHE, SV_FOLDERNAME, USV_FOLDERNAME

print('GT_GLOB   =', const.GT_GLOB_ALL)
print('SV_GLOB   =', const.SV_GLOB_ALL)
print('USV_GLOB  =', const.USV_GLOB_ALL)
print('CACHE     =', const.CACHE)

# Find first number in filename
path_number = lambda path: findall('\d+', basename(path))[0]
sv_imname  = lambda impath: 'image{}.png'.format(path_number(impath))
usv_imname = lambda impath: 'output_image{}.png'.format(path_number(impath))
sv_path  = lambda gt_path: join(CACHE.path, SV_FOLDERNAME, sv_imname(gt_path))
usv_path = lambda gt_path: join(CACHE.path, USV_FOLDERNAME, usv_imname(gt_path))

gt_glob = join(CACHE.path, const.GT_IMG_GLOB)
gt  = imread_collection(gt_glob)
sv_glob = [sv_path(path) for path in gt.files]
usv_glob = [usv_path(path) for path in gt.files]

sv  = imread_collection(sv_glob)
usv = imread_collection(usv_glob)

set_size = np.size(gt.files)

# PLOT
# from matplotlib import pyplot
# from skimage.io import imshow
# imshow(gt[0])
# print(gt[0])
# pyplot.show()

class SamplerTransformer(BaseEstimator, TransformerMixin):
    """ Sample size set to the class minority by default.
        Can specify custom sample size. """
    def __init__(self, sample_size=None):
        self.sample_size = sample_size

    def fit(self, X, y = None):
        return self

    def sample(self, vector):
        """
        Resamples classes to match the size of the smallest class,
        aka the minority.
        """
        # compute vector classes with `np.unique`
        classes, counts = np.unique(vector, return_counts=True)
        assert not classes.size > 2, 'sampler expects no more than 2 classes'
        if (classes.size == 0 or classes.size == 1):
            return [] # e.g. when no road markings

        # split vector by its classes
        splitted = [np.where(vector == classname)[0] for classname in classes]

        # resample. balance equally, to sample_size or to minority size.
        # > note: resample with stratify is sklearn >= v0.21.2
        minority = np.argmin(counts)
        n_samples = self.sample_size or counts[minority]
        resampled = [resample(split, n_samples=n_samples, random_state=41)
                for split in splitted]

        return np.array(resampled).ravel()
    
    def transform(self, X, y = None):
        return [self.sample(vector) for vector in tqdm(X, desc='Sampling')]

def im2vec(im):
    """ Flatten 2D image to a 1D vector. Note that `ravel` creates a view,
    possibly distorting the original array. """
    return np.array(im).ravel()

def ic2vecs(ic):
    """ Maps images in collection to a 1D vector. """
    return [im2vec(im) for im in tqdm(ic, desc='Vectorizing')]

def select_ic(X):
    """ Select certain indices from  """
    data, samples = X
    return [vector[indexes] for vector, indexes in zip(data, samples)]

# kf = KFold(n_splits=2)
# for train_index, test_index in kf.split(gt):
#     print("TRAIN:", train_index, "TEST:", test_index)

# Vectorize
vectorize = FunctionTransformer(ic2vecs, validate=False)
gt  = vectorize.fit_transform(gt)  # replace because using `ravel`
sv  = vectorize.fit_transform(sv)  # replace because using `ravel`
usv = vectorize.fit_transform(usv) # replace because using `ravel`

# Sample
sampler = SamplerTransformer(sample_size=const.SAMPLES)
samples = sampler.fit_transform(gt) # use gt to get sample; classes balanced.

# Select samples
selector = FunctionTransformer(select_ic, validate=False)
gt  = selector.fit_transform((gt, samples))
sv  = selector.fit_transform((sv, samples))
usv = selector.fit_transform((usv, samples))

# Combine into 1D arrays
print('Stacking arrays...')
X = np.stack((np.hstack(sv), np.hstack(usv)), axis=-1)
y = np.hstack(gt)
print('Arrays stacked.')

# Save picklefile
picklepath = '{}_n={}.pickle'.format(const.CACHE.path, set_size)
with open(picklepath, 'wb') as handle:
    pickle.dump((X, y), handle, protocol=pickle.HIGHEST_PROTOCOL)

print('end')