#%%
from os.path import exists, join

import numpy as np
from joblib import dump, load
from matplotlib import pyplot, gridspec
from scipy import stats
from skimage.io import imread, imshow

from constants import (CACHES, DATA_PATH, DUMP_TESTED, GT_FOLDERNAME,
                       GT_IMAGENAME, OUT_FOLDERNAME, SV_FOLDERNAME,
                       SV_IMAGENAME, USV_FOLDERNAME, USV_IMAGENAME,
                       VISUALS_FOLDERPATH)


def plot_prediction_img_comparison():
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    # Read images
    gt = imread(join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png'))
    sv = imread(join(DATA_PATH, SV_FOLDERNAME, SV_IMAGENAME + '1.png'))
    usv = imread(join(DATA_PATH, USV_FOLDERNAME, USV_IMAGENAME + '1.png'))
    cachepath = './tested/cache_100x200/max_samples=200,folds=10,clf=SVM,output'
    out = imread(join(cachepath, GT_IMAGENAME + '1.png'))

    fig, _ = pyplot.subplots(2, 2)
    fig.set_figheight(7)
    fig.subplots_adjust(wspace=0, hspace=0.2)

    # Plot images
    pyplot.subplot(2, 2, 1).set_title("Supervised")
    pyplot.imshow(sv, cmap='gray')
    pyplot.subplot(2, 2, 2).set_title("Unsupervised")
    pyplot.text(40, 125, 'contrast stretched', style='italic',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':3}, fontsize=10,
        color='white')
    pyplot.imshow(usv, cmap='gray')
    pyplot.subplot(2, 2, 3).set_title("Groundtruth")
    pyplot.imshow(gt)
    pyplot.subplot(2, 2, 4).set_title("Prediction")
    pyplot.imshow(out, cmap='gray')
    
    # Save
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, 'prediction-comparison.svg'))


def plot_gt_histogram():
    """ Plot groundtruth image and its associated histogram. """
    # Read
    path = join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png')
    im = imread(path, as_gray=True)
    
    # 2-col plot
    fig, ax = pyplot.subplots(1, 2)

    # Image
    pyplot.subplot(1, 2, 1).set_title("Groundtruth image")
    imshow(im)

    # Histogram
    ax_hist = pyplot.subplot(1, 2, 2)
    ax_hist.set_title("Histogram in 6-bit bins")
    ax_hist.hist(im.ravel(), bins=64, log=True)
    
    # Save
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, 'groundtruth-histogram.svg'))

# def plot_confusion_matrix():


def plot_overall_performance():
    """ Compare cache performance by plotting several boxplots, resembling
        mean fold accuracies. """

    labels = []
    per_cache_means = []
    per_cache_accuracies = []
    all_accuracies = []

    for cache in CACHES:
        cachepath = cache.path.replace('./', './tested/')
        path = join(cachepath, DUMP_TESTED)
        if not exists(path): # skip when cache not tested yet.
            continue

        # Load data from dumpfile
        folded_dataset = load(path)
        folds = folded_dataset['folds']
        
        # accuracy distribution
        fold_mean_accuracies = []
        cache_accuracies = []
        for fold in folds:
            fold_accuracies = fold['accuracies']
            all_accuracies.extend(fold_accuracies)
            cache_accuracies.extend(fold_accuracies)
            fold_mean_accuracies.append(np.mean(fold_accuracies))
        per_cache_means.append(fold_mean_accuracies)
        per_cache_accuracies.append(cache_accuracies)

        # Attach pixel configuration label
        h, w = cache.shape
        labels.append('{}x{}'.format(w, h))
    
    if len(all_accuracies) == 0: # Nothing to plot
        print('No boxplot plotted! - no data for current config found!')
        return

    ##### 2- row plot of accuracy distribution
    fig, ax = pyplot.subplots(2, 1)
    fig.set_figheight(7)
    # (1) Violin plot
    ax_viol = pyplot.subplot(2, 1, 1)
    ax_viol.set_title('Results for {}'.format(OUT_FOLDERNAME))
    ax_viol.set_xlabel('(width x height) in pixels')
    ax_viol.set_ylabel('Accuracy score')

    ax_viol.violinplot(per_cache_accuracies,
        showmeans=True)
    ax_viol.set_xticks(np.arange(1, len(labels) + 1))
    ax_viol.set_xticklabels(labels)
    ax_viol.set_yticks(np.arange(0, 1.1, step=0.1))

    # (2) Accuracy distribution
    ax_hist = pyplot.subplot(2, 1, 2)
    ax_hist.set_title("Accuracy score distribution")
    _, x, _ = ax_hist.hist(all_accuracies, bins=32, density=True, log=True)
    density = stats.gaussian_kde(all_accuracies)
    ax_hist.plot(x, density(x))

    # Save
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-accuracy-distribution.svg'
        .format(OUT_FOLDERNAME)))
    
    ##### Boxplot with fold means
    fig, ax = pyplot.subplots()
    ax.set_xlabel('(width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Results for {}'.format(OUT_FOLDERNAME))
    ax.boxplot(per_cache_means, labels=labels)

    # Save
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-boxplot.svg'
        .format(OUT_FOLDERNAME)))

plot_prediction_img_comparison()
plot_gt_histogram()
plot_overall_performance()
