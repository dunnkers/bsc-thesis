#%%
import pickle
from os import makedirs
from os.path import dirname, exists, join
from time import time
from warnings import catch_warnings, simplefilter

import numpy as np
from matplotlib import pyplot
from skimage.io import imread_collection, imsave, imshow
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm

from constants import (CACHES, GT_FOLDERNAME, OUT_FOLDERNAME,
                       PICKLEFILE_PREPARED)


def makeDirIfNotExists(impath):
    if not exists(dirname(impath)):
        makedirs(dirname(impath))

def test_fold(fold, cache, gt_files):
    _, _, X_test, y_test = fold['data']
    clf = fold['clf']
    test_indexes = fold['test_indexes']
    test_size = test_indexes.size
    
    # Predict
    X_test_images = np.split(X_test, test_size) 
    y_test_images = np.split(y_test, test_size)
    assert(X_test.shape[0] == y_test.shape[0])

    for i, X_test_img in enumerate(X_test_images):
        w, h = cache.shape
        print('[{}/{}] Predicting {}x{} image...'
            .format(i + 1, test_size, w, h), end='\t')
        start = time()
        predictions = clf.predict(X_test_img) # ⚠️ Only start using terminal.
        # predictions = np.zeros(np.prod(cache.shape))

        # Accuracy score
        acc_score = accuracy_score(y_test_images[i], predictions)
        end = time()
        print('predicted in {0:.2f} sec; accuracy = {1:.2f}%'
            .format(end - start, acc_score * 100))

        # Reconstruction
        impath_gt  = gt_files[test_indexes[i]]
        impath_out = impath_gt.replace(GT_FOLDERNAME, OUT_FOLDERNAME)
        makeDirIfNotExists(impath_out)
        im = np.reshape(predictions, cache.shape)

        # Save
        # imshow(im)
        # pyplot.show()
        with catch_warnings(): # prevent contrast warnings.
            simplefilter("ignore")
            imsave(impath_out, im, check_contrast=False)

        break # 🛑 only test one image for now.

def test_cache(cache):
    picklepath = join(cache.path, PICKLEFILE_PREPARED)
    with open(picklepath, 'rb') as handle:
        folded_dataset = pickle.load(handle)
        n_splits = folded_dataset['n_splits']
        gt_files = folded_dataset['gt_files']

        # Test every fold
        for i, fold in enumerate(folded_dataset['folds']):
            print('[{}/{}] Testing fold...'.format(i + 1, n_splits))
            test_fold(fold, cache, gt_files)

def test_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Testing cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        test_cache(cache)

test_all()
print('Finished testing.')
