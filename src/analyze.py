#%%
from os.path import basename, exists, join

import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from joblib import dump, load
from matplotlib import gridspec, pyplot as plt
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.util import img_as_bool, img_as_ubyte, img_as_float
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from constants import (CACHES, CONFIG_STR, CONFIG_STR_NOCLF, DATA_PATH,
                       DUMP_TESTED, GT_DATA_GLOB, GT_FOLDERNAME, GT_IMAGENAME,
                       IMG_GLOB, N_FOLDS, OUT_FOLDERNAME, SV_FOLDERNAME,
                       SV_IMAGENAME, USV_FOLDERNAME, USV_IMAGENAME,
                       VISUALS_CONFIG_STR, VISUALS_FOLDERPATH)


def compute_and_plot_AUCROC(y_true, y_score, title='AUC/ROC Curve'):
    print('Computing AUC score / ROC curve...')
    start = time()

    # calculate AUC
    auc = roc_auc_score(y_true, y_score)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    end = time()
    print('Computation took %.2f seconds' % end - start)

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title(title)
    # show the plot
    plt.show()

def im2binary(im):
    return np.array(img_as_float(im)).astype('int')

def ic2binary(ic):
    return np.array([im2binary(im) for im in ic]).ravel()

def ic2probabilities(ic):
    # (1) Read in output images
    # (2) Convert to [0, 1] range
    # (3) Convert to 1D array
    return np.array([img_as_float(im) for im in ic]).ravel()

def compute_probabilities(cachepath, out_foldername):
    # Read ground truth images.
    gt_glob = join(cachepath, GT_FOLDERNAME, IMG_GLOB)
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
    compute_and_plot_AUCROC(gt_bin, sv_prob, title='Supervised ROC/AUC')

    # unsupervised
    usv_prob = ic2probabilities(usv)
    compute_and_plot_AUCROC(gt_bin, usv_prob, title='Unsupervised ROC/AUC')

    # fusion
    # (1) Read in trained classifier
    # (2) Use `predict_proba`
    pass

def compute_all_probabilities():
    for cache in CACHES:
        compute_probabilities(cache, OUT_FOLDERNAME)


compute_probabilities('./cache_100x200', OUT_FOLDERNAME)