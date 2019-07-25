#%%
from os.path import exists, join

import numpy as np
from joblib import dump, load
from matplotlib import gridspec, pyplot
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.util import img_as_bool, img_as_ubyte
from tqdm.auto import tqdm

from constants import (CACHES, CONFIG_STR, DATA_PATH, DUMP_TESTED,
                       GT_DATA_GLOB, GT_FOLDERNAME, GT_IMAGENAME,
                       OUT_FOLDERNAME, SV_FOLDERNAME, SV_IMAGENAME,
                       USV_FOLDERNAME, USV_IMAGENAME, VISUALS_CONFIG_STR,
                       VISUALS_FOLDERPATH)


def plot_prediction_img_comparison():
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    cachepath = './cache_140x280'
    cachepath = cachepath.replace('./', './tested/')
    impath = join(cachepath, OUT_FOLDERNAME, '{}1.png'.format(GT_IMAGENAME))
    if not exists(impath): # skip when cache not tested yet.
        return

    # Read images
    gt = imread(join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png'))
    sv = imread(join(DATA_PATH, SV_FOLDERNAME, SV_IMAGENAME + '1.png'))
    usv = imread(join(DATA_PATH, USV_FOLDERNAME, USV_IMAGENAME + '1.png'))
    out = imread(impath)

    fig, _ = pyplot.subplots(2, 2)
    fig.set_figheight(7)
    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.suptitle('{}'
        .format(VISUALS_CONFIG_STR))

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
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-comparison.svg'
        .format(CONFIG_STR)))


def plot_gt_histogram():
    """ Plot groundtruth image and its associated histogram. """
    # Read
    path = join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png')
    im = imread(path, as_gray=True)
    
    # 2-col plot
    fig, _ = pyplot.subplots(1, 2)

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

    ##### Violin plot - accuracy performance & distribution
    fig, ax = pyplot.subplots()
    ax.set_title('{}'
        .format(VISUALS_CONFIG_STR))
    ax.set_xlabel('(width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.violinplot(per_cache_accuracies,
        showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    fig.autofmt_xdate()
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-violinplot.svg'
        .format(CONFIG_STR)))

    ##### Histogram - accuracy distribution
    fig, ax = pyplot.subplots()
    ax.set_title("Accuracy score distribution")
    _, x, _ = ax.hist(all_accuracies, bins=32, density=True, log=True)
    density = stats.gaussian_kde(all_accuracies)
    ax.plot(x, density(x))
    # ax.set_xticks(np.arange(1, 1.1, step=0.1)) # @FIXME ticks with 0.1 steps
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-histogram.svg'
        .format(CONFIG_STR)))
    
    ##### Boxplot with fold means
    fig, ax = pyplot.subplots()
    ax.set_xlabel('(width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.set_title('{}'
        .format(VISUALS_CONFIG_STR))
    fig.autofmt_xdate()
    ax.boxplot(per_cache_means, labels=labels)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-boxplot.svg'
        .format(CONFIG_STR)))

# def plot_confusion_matrix():

def plot_acc_vs_gt_fractions():
    cachepath = './cache_140x280'
    gt  = imread_collection(join(cachepath, 'groundtruth/*.png'))
    gt_fractions = []
    for i in tqdm(range(len(gt.files)), desc="Computing gt fractions"):
        gtimg = gt[i]
        classes, counts = np.unique(gtimg, return_counts=True)
        if len(classes) == 0:
            continue # image error
        elif len(classes) == 1:
            fraction = 0 # no road markings at all. fraction = 0
        else: # > 1
            road, road_marker = counts
            fraction = road_marker / road
        gt_fractions.append(fraction)
    
    ####### Accuracies
    gt_accs = np.zeros(len(gt.files))

    cachepath_tested = cachepath.replace('./', './tested/')
    path = join(cachepath_tested, DUMP_TESTED)
    if not exists(path): # skip when cache not tested yet.
        return

    # Load data from dumpfile
    folded_dataset = load(path)
    folds = folded_dataset['folds']
    
    # accuracy distribution
    for fold in folds:
        accuracies = fold['accuracies']
        test_indexes = fold['test_indexes']

        for i in range(len(test_indexes)):
            gt_accs[test_indexes[i]] = accuracies[i]

    fig, ax = pyplot.subplots()
    ax.scatter(gt_fractions, gt_accs, marker='.', alpha=0.3)
    ax.set_title('Accuracy score vs. fraction of pixels of type road marking')
    ax.set_xlabel('Fraction of pixels of type road marking')
    ax.set_ylabel('Accuracy score')

    # Save
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-scatterplot.svg'
        .format(CONFIG_STR)))

plot_prediction_img_comparison()
plot_gt_histogram()
plot_overall_performance()
plot_acc_vs_gt_fractions()
