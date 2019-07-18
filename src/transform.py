#%%
import pickle
from os.path import basename, join
from re import findall

import numpy as np
from skimage.io import imread_collection
from skimage.util import img_as_bool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample
from tqdm.auto import tqdm

from constants import (CACHES, GT_FOLDERNAME, GT_IMAGENAME, IMG_GLOB,
                       MAX_SAMPLES, N_FOLDS, PICKLEFILE_PREPARED,
                       SV_FOLDERNAME, SV_IMAGENAME, USV_FOLDERNAME,
                       USV_IMAGENAME)

print('N_FOLDS =', N_FOLDS)

# @IDEA: Find out where Sampler(stratified=True) and Stratified k-fold comes in.
class SamplerTransformer(BaseEstimator, TransformerMixin):
    """ Sample size set to the class minority by default.
        Can specify custom sample size. Balanced sampler. """
    def __init__(self, max_sample_size=None):
        self.max_sample_size = max_sample_size

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

        # resample. balance equally, to max_sample_size or to minority size.
        # > note: resample with stratify is sklearn >= v0.21.2
        minority = np.argmin(counts)
        n_samples = self.max_sample_size or counts[minority]
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
    """ Map each image in collection to select the given indices. """
    data, samples = X
    return [vector[indexes] for vector, indexes in zip(data, samples)]

# def prepare_fold(fold):
#   pass

def prepare_cache(cache):
    """ Prepare images in cache folder. Transforms 2D image arrays into flat
    arrays, samples the images using balanced classes, and combines supervised-
    and unsupervised approaches into a 2-feature vector. """
    # Read ground truth images.
    gt_glob = join(cache.path, GT_FOLDERNAME, IMG_GLOB)
    gt = imread_collection(gt_glob)

    # Then, map sv/usv images for every gt image.
    sv_glob  = [path.replace(GT_FOLDERNAME, SV_FOLDERNAME)
                    .replace(GT_IMAGENAME, SV_IMAGENAME)  for path in gt.files]
    usv_glob = [path.replace(GT_FOLDERNAME, USV_FOLDERNAME)
                    .replace(GT_IMAGENAME, USV_IMAGENAME) for path in gt.files]

    # Read sv/usv
    sv  = imread_collection(sv_glob)
    usv = imread_collection(usv_glob)

    # Size
    assert(np.size(gt.files) == np.size(sv.files) == np.size(usv.files))

    # Split the set
    kf = KFold(n_splits=N_FOLDS)
    folded_dataset = { # @IDEA use a Dataset class for this.
        'size': np.size(gt.files),
        'folds': [],
        'max_samples': MAX_SAMPLES,
        'n_splits': N_FOLDS,
        'cachepath': cache.path,
        'shape': cache.shape
    }

    # Instantiate transformers
    vectorize = FunctionTransformer(ic2vecs, validate=False)
    sampler = SamplerTransformer(max_sample_size=MAX_SAMPLES)
    selector = FunctionTransformer(select_ic, validate=False)

    # Convert image collections to array
    gt_arr  = np.array(gt)
    sv_arr  = np.array(sv)
    usv_arr = np.array(usv)

    for train_index, test_index in kf.split(gt):
        print('[{}/{}] Building dataset fold of size {}...'
            .format(len(folded_dataset['folds']) + 1, N_FOLDS, train_index.size))
        assert(train_index.size == test_index.size) # just making sure
        gt_train, gt_test   = gt_arr[train_index],  gt_arr[test_index]
        sv_train, sv_test   = sv_arr[train_index],  sv_arr[test_index]
        usv_train, usv_test = usv_arr[train_index], usv_arr[test_index]

        ### TRAINING
        print('Building train data...')

        # Vectorize
        gt_train  = vectorize.fit_transform(gt_train)
        sv_train  = vectorize.fit_transform(sv_train)
        usv_train = vectorize.fit_transform(usv_train)

        # Sample
        samples_train = sampler.fit_transform(gt_train)

        # Select samples
        gt_train  = selector.fit_transform((gt_train, samples_train))
        sv_train  = selector.fit_transform((sv_train, samples_train))
        usv_train = selector.fit_transform((usv_train, samples_train))

        # Combine into 1D arrays
        X_train = np.stack((np.hstack(sv_train), np.hstack(usv_train)), axis=-1)
        y_train = np.hstack(gt_train)


        ### TESTING
        print('Building test data...')

        # Vectorize
        gt_test  = vectorize.transform(gt_test)
        sv_test  = vectorize.transform(sv_test)
        usv_test = vectorize.transform(usv_test)

        # Combine into 1D arrays
        X_test = np.stack((np.hstack(sv_test), np.hstack(usv_test)), axis=-1)
        y_test = np.hstack(gt_test)

        # Add to folded datasets
        fold = dict(data=(X_train, y_train, X_test, y_test),
                    size=train_index.size)
        folded_dataset['folds'].append(fold)

    picklepath = join(cache.path, PICKLEFILE_PREPARED)
    with open(picklepath, 'wb') as handle:
        pickle.dump(folded_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Preparing cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        prepare_cache(cache)

prepare_all()
print('Finished dataset preparation.')
