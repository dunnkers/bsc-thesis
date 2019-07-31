#%%
from os.path import basename, exists, join
from time import time

import numpy as np
import pandas as pd
from os import makedirs
from os.path import basename, dirname, exists, join
import seaborn as sns
from joblib import dump, load
from warnings import catch_warnings, simplefilter
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread_collection, imsave, imshow
from skimage.util import img_as_bool, img_as_float, img_as_ubyte
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from constants import (CACHES, CONFIG_STR, CONFIG_STR_NOCLF, DATA_PATH,
                       DUMP_TESTED, DUMP_TRAINED, GT_DATA_GLOB, GT_FOLDERNAME,
                       GT_IMAGENAME, IMG_GLOB, N_FOLDS, OUT_FOLDERNAME,
                       SV_FOLDERNAME, SV_IMAGENAME, USV_FOLDERNAME,
                       USV_IMAGENAME, VISUALS_CONFIG_STR, VISUALS_FOLDERPATH,
                       DUMP_TRANSFORMED, Cache)


def compute_and_plot_AUCROC(y_true, y_score, ic_name, title='AUC/ROC Curve'):
    print('Computing {} AUC score / ROC curve...'.format(ic_name))
    start = time()

    # calculate AUC
    auc = roc_auc_score(y_true, y_score)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    end = time()
    print('Computation took %.2f seconds' % (end - start))

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title(title)
    plt.xlabel('AUC score: {:.3f}'.format(auc))

    # Save
    plt.savefig(join(VISUALS_FOLDERPATH, '{}-aucroc.svg'
        .format(ic_name)))
    plt.clf()

def im2binary(im):
    return np.array(img_as_float(im)).astype('int')

def ic2binary(ic):
    return np.array([im2binary(im) for im in ic]).ravel()

def ic2probabilities(ic):
    return np.array([img_as_float(im) for im in ic]).ravel()

def makeDirIfNotExists(impath):
    if not exists(dirname(impath)):
        makedirs(dirname(impath))

def predict_image_prob(clf, X, y, shape, impath_gt):
    # Predict
    probabilities = clf.predict_proba(X)
    print('both classes',probabilities)
    probabilities = probabilities[:, 1]
    print('positive class only',probabilities)
    # Accuracy
    accuracy = 0

    # Reconstruction
    im = np.reshape(probabilities, shape)

    # Output
    impath_out = impath_gt.replace(GT_FOLDERNAME, OUT_FOLDERNAME + '_proba')
    makeDirIfNotExists(impath_out)
    with catch_warnings(): # prevent contrast warnings.
        simplefilter("ignore")
        imsave(impath_out, im, check_contrast=False)
    
    return accuracy

def predict_fold_prob(fold, clf, cache, gt_files, i):
    _, _, X_test, y_test = fold['data']
    test_indexes = fold['test_indexes']
    test_size = test_indexes.size
    
    # Predict
    assert(len(X_test) == len(y_test))
    X_test_imgs = np.split(X_test, test_size) 
    y_test_imgs = np.split(y_test, test_size)
    
    accuracies = []
    for i in tqdm(range(test_size),
        desc=' Analyzing fold  {}'.format(i + 1), unit='img'):
        X = X_test_imgs[i]
        y = y_test_imgs[i]
        gt_idx = test_indexes[i]
        impath_gt = gt_files[gt_idx]
        
        accuracy = predict_image_prob(clf, X, y, cache.shape, impath_gt)
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    return accuracies

def predict_cache_prob(cache):
    folded_dataset = load(join(cache.path, DUMP_TRANSFORMED))
    clfs = load(join(cache.path, DUMP_TRAINED))
    
    n_splits = folded_dataset['n_splits']
    gt_files = folded_dataset['gt_files']
    folds    = folded_dataset['folds']

    # Test every fold
    for i in tqdm(range(n_splits), 
        desc=' Analyzing {} folds'.format(n_splits), unit='fold'):
        clf = clfs[i]
        fold = folds[i]
        fold_accuracies = predict_fold_prob(fold, clf, cache, gt_files, i)

        # Save accuracy
        fold['accuracies'] = fold_accuracies

        # Detach data from fold: do not dump data.
        del fold['data']
    ### METHOD A
    # there are 10 trained classifiers. pick the best performing one to compute
    # its probabilities.
    # (1) do an image reconstruction. whats it look like?
    # (2) compute probabilities.

    ### METHOD B
    # (1) let every fold's trained classifier predict probabilities for its specified
    # test indexes, just like normal.
    # (2) store all probability scores.
    # (3) compute auc/roc


    # (1) Read in trained classifier
    # (2) Use `predict_proba`
    return []


def compute_probabilities(cache, out_foldername):
    # Read ground truth images.
    gt_glob = join(cache.path, GT_FOLDERNAME, IMG_GLOB)
    gt = imread_collection(gt_glob)

    # Then, map sv/usv images for every gt image.
    sv_glob  = [path.replace(GT_FOLDERNAME, SV_FOLDERNAME)
                    .replace(GT_IMAGENAME,  SV_IMAGENAME)  for path in gt.files]
    usv_glob = [path.replace(GT_FOLDERNAME, USV_FOLDERNAME)
                    .replace(GT_IMAGENAME,  USV_IMAGENAME) for path in gt.files]
    # Read sv/usv
    sv  = imread_collection(sv_glob)
    usv = imread_collection(usv_glob)

    # groundtruth
    gt_bin   = ic2binary(gt)

    # supervised
    sv_prob  = ic2probabilities(sv)
    compute_and_plot_AUCROC(gt_bin, sv_prob, 'supervised', 
        title='Supervised AUC score/ROC curve')

    # unsupervised
    usv_prob = ic2probabilities(usv)
    compute_and_plot_AUCROC(gt_bin, usv_prob, 'unsupervised', 
        title='Unsupervised AUC score/ROC curve')

    # # fusion
    # fsd_prob = predict_cache_prob(cache)
    # compute_and_plot_AUCROC(gt_bin, fsd_prob, 'fusion', 
    #     title='Fusion AUC score/ROC curve')

def compute_all_probabilities():
    for cache in CACHES:
        compute_probabilities(cache, OUT_FOLDERNAME)

# Compute fusion probabilities
predict_cache_prob(Cache('./cache_100x200',  (200, 100)))
# compute_probabilities(Cache('./cache_100x200',  (200, 100)), OUT_FOLDERNAME)


#%%
