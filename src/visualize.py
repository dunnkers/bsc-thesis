#%%
from os.path import exists, join

import numpy as np
from joblib import dump, load
from matplotlib import pyplot
from skimage.io import imread, imshow

from constants import (CACHES, DATA_PATH, DUMP_TESTED, GT_FOLDERNAME,
                       GT_IMAGENAME, OUT_FOLDERNAME, SV_FOLDERNAME,
                       SV_IMAGENAME, USV_FOLDERNAME, USV_IMAGENAME,
                       VISUALS_FOLDERPATH)


def plot_comparison():
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    # Read images
    gt = imread(join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png'))
    sv = imread(join(DATA_PATH, SV_FOLDERNAME, SV_IMAGENAME + '1.png'))
    usv = imread(join(DATA_PATH, USV_FOLDERNAME, USV_IMAGENAME + '1.png'))
    cachepath = './cache_175x350/filtered,output_max_samples=100,folds=5,clf=XGBoost'
    out = imread(join(cachepath, GT_IMAGENAME + '1.png'))

    # Plot images
    pyplot.subplot(1, 4, 1).set_title("Groundtruth")
    imshow(gt)
    pyplot.subplot(1, 4, 2).set_title("Supervised")
    imshow(sv)
    pyplot.subplot(1, 4, 3).set_title("Unsupervised")
    imshow(usv)
    pyplot.subplot(1, 4, 4).set_title("Prediction")
    imshow(out)
    
    # Save
    pyplot.tight_layout()
    pyplot.savefig(join(VISUALS_FOLDERPATH, 'prediction-comparison.svg'))


def plot_gt_histogram():
    """ Plot groundtruth image and its associated histogram. """
    # Read
    path = join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png')
    im = imread(path, as_gray=True)

    # Image
    pyplot.subplot(1, 2, 1).set_title("Groundtruth image")
    imshow(im)

    # Histogram
    ax_hist = pyplot.subplot(1, 2, 2)
    ax_hist.set_title("Histogram in 6-bit bins")
    ax_hist.hist(im.ravel(), bins=64, log=True)
    
    # Save
    pyplot.tight_layout()
    pyplot.savefig(join(VISUALS_FOLDERPATH, 'groundtruth-histogram.svg'))

def plot_boxplot():
    """ Compare cache performance by plotting several boxplots, resembling
        mean fold accuracies. """
    data = []
    labels = []

    for cache in CACHES:
        path = join(cache.path, DUMP_TESTED)
        if not exists(path): # skip when cache not tested yet.
            continue

        # Load data from dumpfile
        results = load(path)
        accuracies = results['accuracies']
        
        # Transform such that we can plot
        cache_data = list(map(np.average, accuracies))
        data.append(cache_data)
        h, w = cache.shape
        labels.append('{}x{}'.format(w, h))
    
    if len(data) == 0: # Nothing to plot
        print('No boxplot plotted! - no data for current config found!')
        return
    
    # Boxplot
    _, ax = pyplot.subplots()
    ax.set_xlabel('(width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Results for {}'.format(OUT_FOLDERNAME))
    ax.boxplot(data, labels=labels)

    # Save
    pyplot.savefig(join(VISUALS_FOLDERPATH, '{}-boxplot.svg'
        .format(OUT_FOLDERNAME)))

plot_comparison()
plot_gt_histogram()
plot_boxplot()
