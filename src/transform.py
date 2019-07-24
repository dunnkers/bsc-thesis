#%%
from datetime import timedelta
from os.path import basename, join
from re import findall
from time import time

import numpy as np
from joblib import dump, load
from skimage.io import imread_collection
from skimage.util import img_as_bool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample
from tqdm.auto import tqdm

from constants import (CACHES, DUMP_TRANSFORMED, GT_FOLDERNAME, GT_IMAGENAME,
                       IMG_GLOB, MAX_SAMPLES, N_FOLDS, SV_FOLDERNAME,
                       SV_IMAGENAME, USV_FOLDERNAME, USV_IMAGENAME)

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
        assert(self.max_sample_size % 2 == 0) # make sure is divisible by 2.
        n_samples = int(self.max_sample_size / 2) or counts[minority]
        resampled = [resample(split, n_samples=n_samples, random_state=41)
                for split in splitted]

        return np.array(resampled).ravel()
    
    def transform(self, X, y = None):
        return [self.sample(vector) for vector in tqdm(X, desc='  Sampling')]

def im2vec(im):
    """ Flatten 2D image to a 1D vector. Note that `ravel` creates a view,
    possibly distorting the original array. """
    return np.array(im).ravel()

def ic2vecs(ic):
    """ Maps images in collection to a 1D vector. """
    return [im2vec(im) for im in ic]

def select_ic(X):
    """ Map each image in collection to select the given indices. """
    data, samples = X
    return [vector[indexes] for vector, indexes in zip(data, samples)]

def transform_fold(transformers, split, gt_arr, sv_arr, usv_arr):
    train_indexes, test_indexes = split
    gt_train, gt_test   = gt_arr[train_indexes],  gt_arr[test_indexes]
    sv_train, sv_test   = sv_arr[train_indexes],  sv_arr[test_indexes]
    usv_train, usv_test = usv_arr[train_indexes], usv_arr[test_indexes]

    ### TRAINING
    # Vectorize
    gt_train  = transformers['vectorize'].fit_transform(gt_train)
    sv_train  = transformers['vectorize'].fit_transform(sv_train)
    usv_train = transformers['vectorize'].fit_transform(usv_train)

    # Sample
    samples_train = transformers['sampler'].fit_transform(gt_train)

    # Select samples
    gt_train  = transformers['selector'].fit_transform((gt_train, samples_train))
    sv_train  = transformers['selector'].fit_transform((sv_train, samples_train))
    usv_train = transformers['selector'].fit_transform((usv_train, samples_train))

    # Combine into 1D arrays
    X_train = np.stack((np.hstack(sv_train), np.hstack(usv_train)), axis=-1)
    y_train = np.hstack(gt_train)


    ### TESTING
    # Vectorize
    gt_test  = transformers['vectorize'].transform(gt_test)
    sv_test  = transformers['vectorize'].transform(sv_test)
    usv_test = transformers['vectorize'].transform(usv_test)

    # Combine into 1D arrays
    X_test = np.stack((np.hstack(sv_test), np.hstack(usv_test)), axis=-1)
    y_test = np.hstack(gt_test)

    # Construct fold
    fold = dict(data=(X_train, y_train, X_test, y_test),
                train_indexes=train_indexes,
                test_indexes=test_indexes)
    return fold

def transform_cache(cache):
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

    # Assert size
    assert(np.size(gt.files) == np.size(sv.files) == np.size(usv.files))

    # Instantiate transformers
    vectorize = FunctionTransformer(ic2vecs, validate=False)
    sampler = SamplerTransformer(max_sample_size=MAX_SAMPLES)
    selector = FunctionTransformer(select_ic, validate=False)
    transformers = dict(vectorize=vectorize, sampler=sampler, selector=selector)

    # Convert image collections to array
    print(' Converting image collections to arrays...')
    start = time()
    gt_arr  = np.array(gt)
    sv_arr  = np.array(sv)
    usv_arr = np.array(usv)
    end = time()
    print(' Converted to arrays in {}'.format(timedelta(seconds=end - start)))

    # Split the set
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=8)

    # Transform each fold
    folds = []
    for i, split in enumerate(kf.split(gt)):
        print(' [{}/{}] Building dataset fold split {}/{}...'
            .format(i + 1, N_FOLDS, split[0].size, split[1].size))
        fold = transform_fold(transformers, split, gt_arr, sv_arr, usv_arr)
        folds.append(fold)

    # Construct folded dataset
    folded_dataset = dict(folds=folds,
                          max_samples=MAX_SAMPLES,
                          n_splits=N_FOLDS,
                          gt_files=gt.files)

    # Dump to file
    dump(folded_dataset, join(cache.path, DUMP_TRANSFORMED))

def transform_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Transforming cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        start = time()
        transform_cache(cache)
        end = time()
        print('Transformed cache in {}'.format(timedelta(seconds=end - start)))

start = time()
transform_all()
end = time()
print('Finished transforming in {}'.format(timedelta(seconds=end - start)))
