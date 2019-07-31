#%%
from os.path import basename, exists, join
from time import time

import numpy as np
from datetime import timedelta
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
                       DUMP_TRANSFORMED, Cache, PROBA_FOLDERNAME)


def compute_and_plot_ic_AUCROC(y_true, y_score, ic_name, title='AUC/ROC Curve'):
    print(' Computing {} AUC score / ROC curve...'.format(ic_name))
    start = time()

    # calculate AUC
    auc = roc_auc_score(y_true, y_score)
    print('  AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    end = time()
    print('  Computed in {}'.format(timedelta(seconds=end - start)))

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

def compute_and_plot_cache_AUCROC(cache):
    # Read ground truth images.
    gt_glob = join(cache.path, GT_FOLDERNAME, IMG_GLOB)
    gt = imread_collection(gt_glob)

    # Then, map sv/usv images for every gt image.
    sv_glob  = [path.replace(GT_FOLDERNAME, SV_FOLDERNAME)
                    .replace(GT_IMAGENAME,  SV_IMAGENAME)  for path in gt.files]
    usv_glob = [path.replace(GT_FOLDERNAME, USV_FOLDERNAME)
                    .replace(GT_IMAGENAME,  USV_IMAGENAME) for path in gt.files]
    fsd_glob = [path.replace(GT_FOLDERNAME, PROBA_FOLDERNAME)
                                                           for path in gt.files]
    # Read sv/usv
    sv  = imread_collection(sv_glob)
    usv = imread_collection(usv_glob)
    fsd = imread_collection(fsd_glob)

    # groundtruth
    gt_bin   = ic2binary(gt) # in binary: [0, 1]

    # supervised
    sv_prob  = ic2probabilities(sv)
    compute_and_plot_ic_AUCROC(gt_bin, sv_prob, 'supervised', 
        title='Supervised AUC score/ROC curve')

    # unsupervised
    usv_prob = ic2probabilities(usv)
    compute_and_plot_ic_AUCROC(gt_bin, usv_prob, 'unsupervised', 
        title='Unsupervised AUC score/ROC curve')

    # fusion
    fsd_prob = ic2probabilities(fsd)
    compute_and_plot_ic_AUCROC(gt_bin, fsd_prob, 'fusion', 
        title='Fusion AUC score/ROC curve')

def compute_and_plot_all_AUCROC():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Computing cache \'{}\' AUC/ROC...'
            .format(i + 1, len(CACHES), cache.path))
        compute_and_plot_cache_AUCROC(cache)


start = time()
compute_and_plot_all_AUCROC()
end = time()
print('Finished AUC/ROC computation in {}'.format(timedelta(seconds=end - start)))
