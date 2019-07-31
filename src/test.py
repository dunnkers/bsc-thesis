#%%
import pickle
from datetime import timedelta
from os import makedirs
from os.path import basename, dirname, exists, join
from time import time
from warnings import catch_warnings, simplefilter

import numpy as np
from joblib import dump, load
from skimage.io import imread_collection, imsave, imshow
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm

from constants import (CACHES, COMPUTE_CLASS, COMPUTE_PROBA, DUMP_TESTED,
                       DUMP_TRAINED, DUMP_TRANSFORMED, GT_FOLDERNAME,
                       OUT_FOLDERNAME, PROBA_FOLDERNAME)


def makeDirIfNotExists(impath):
    if not exists(dirname(impath)):
        makedirs(dirname(impath))

def predict_image_proba(clf, X, y, shape, impath_gt):
    # Predict
    probabilities = clf.predict_proba(X)
    probabilities = probabilities[:, 1] # Save positive class probability only

    # Reconstruction
    im = np.reshape(probabilities, shape)

    # Output
    impath_out = impath_gt.replace(GT_FOLDERNAME, PROBA_FOLDERNAME)
    makeDirIfNotExists(impath_out)
    with catch_warnings(): # prevent contrast warnings.
        simplefilter("ignore")
        imsave(impath_out, im, check_contrast=False)

def predict_image(clf, X, y, shape, impath_gt):
    # Predict
    predictions = clf.predict(X) # predictions = np.zeros(np.prod(shape)) # (🧪)

    # Accuracy
    accuracy = accuracy_score(y, predictions)

    # Reconstruction
    im = np.reshape(predictions, shape)

    # Output
    impath_out = impath_gt.replace(GT_FOLDERNAME, OUT_FOLDERNAME)
    makeDirIfNotExists(impath_out)
    with catch_warnings(): # prevent contrast warnings.
        simplefilter("ignore")
        imsave(impath_out, im, check_contrast=False)
    
    return accuracy


def test_fold(fold, clf, cache, gt_files, i):
    _, _, X_test, y_test = fold['data']
    test_indexes = fold['test_indexes']
    test_size = test_indexes.size
    
    # Predict
    assert(len(X_test) == len(y_test))
    X_test_imgs = np.split(X_test, test_size) 
    y_test_imgs = np.split(y_test, test_size)
    
    accuracies = []
    for i in tqdm(range(test_size),
        desc=' Testing fold  {}'.format(i + 1), unit='img'):
        X = X_test_imgs[i]
        y = y_test_imgs[i]
        gt_idx = test_indexes[i]
        impath_gt = gt_files[gt_idx]

        if COMPUTE_CLASS:
            accuracy = predict_image(clf, X, y, cache.shape, impath_gt)
            accuracies.append(accuracy)
        
        if COMPUTE_PROBA:
            predict_image_proba(clf, X, y, cache.shape, impath_gt)
    
    accuracies = np.array(accuracies)
    return accuracies

def test_cache(cache):
    folded_dataset = load(join(cache.path, DUMP_TRANSFORMED))
    clfs = load(join(cache.path, DUMP_TRAINED))

    n_splits = folded_dataset['n_splits']
    gt_files = folded_dataset['gt_files']
    folds    = folded_dataset['folds']

    # Test every fold
    for i in tqdm(range(n_splits), 
        desc=' Testing {} folds'.format(n_splits), unit='fold'):
        clf = clfs[i]
        fold = folds[i]
        fold_accuracies = test_fold(fold, clf, cache, gt_files, i)

        # Save accuracy
        fold['accuracies'] = fold_accuracies

        # Detach data from fold: do not dump data.
        del fold['data']
    
    if not COMPUTE_CLASS:
        return

    # Compute mean
    # > fold shapes are not guaranteed to be homogeneous. take into account
    # whilst using numpy functions. some expect homogeneous arrays, like mean().
    means = list(map(
        lambda fold: np.round(np.mean(fold['accuracies']), decimals=4), folds))
    tqdm.write('\nCache accuracies: {}, mean = {:.4f}'
        .format(means, np.mean(means)))
    
    # Dump results
    dumppath = join(cache.path, DUMP_TESTED)
    dump(folded_dataset, dumppath)

    # txtfile
    txtfile = open(dumppath + ".txt", "w")
    txtfile.writelines(str(folded_dataset)) 
    txtfile.close()

def test_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Testing cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        test_cache(cache)

start = time()
test_all()
end = time()
print('Finished testing in {}'.format(timedelta(seconds=end - start)))
