#%%
from os.path import exists, join, basename

import numpy as np
from joblib import dump, load
from matplotlib import gridspec, pyplot
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.util import img_as_bool, img_as_ubyte
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid

from constants import (CACHES, CONFIG_STR, DATA_PATH, DUMP_TESTED,
                       GT_DATA_GLOB, GT_FOLDERNAME, GT_IMAGENAME,
                       OUT_FOLDERNAME, SV_FOLDERNAME, SV_IMAGENAME,
                       USV_FOLDERNAME, USV_IMAGENAME, VISUALS_CONFIG_STR,
                       VISUALS_FOLDERPATH)


def plot_prediction_img_comparison():
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    cachepath = './cache_140x280'
    tested_path = cachepath.replace('./', './tested/')
    impath = join(tested_path, OUT_FOLDERNAME, '{}1.png'.format(GT_IMAGENAME))
    if not exists(impath): # skip when cache not tested yet.
        return

    # Read images
    gt = imread(join(cachepath, GT_FOLDERNAME, GT_IMAGENAME + '1.png'))
    sv = imread(join(cachepath, SV_FOLDERNAME, SV_IMAGENAME + '1.png'))
    usv = imread(join(cachepath, USV_FOLDERNAME, USV_IMAGENAME + '1.png'))
    out = imread(impath)

    # Size
    h, w = gt.shape

    # Plot 2x2
    fig = pyplot.figure(figsize=(7.0, 9.0))
    fig.suptitle('Prediction result')

    grid = ImageGrid(fig, (0.1, 0.1, 0.8, 0.8), 
        nrows_ncols=(2, 2), axes_pad=(0.15, 0.5), label_mode="L", aspect=True)
    grid[0].set_title('Supervised')
    grid[0].imshow(sv, cmap='gray')

    grid[1].set_title('Unsupervised')
    grid[1].imshow(usv, cmap='gray')

    grid[2].set_title('Groundtruth')
    grid[2].imshow(gt, cmap='gray')

    grid[3].set_title('Prediction')
    grid[3].imshow(out, cmap='gray')

    fig.text(0.218, 0.945, '{}, cache={}x{}, im={}'
        .format(VISUALS_CONFIG_STR, w, h, basename(impath)), fontsize=10)

    fig.text(0.54, 0.88, 'contrast stretched', fontsize=10, color='white',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':2})

    fig.savefig(join(VISUALS_FOLDERPATH, '{}-comparison.svg'
        .format(CONFIG_STR)), bbox_inches='tight')


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
        .format(VISUALS_CONFIG_STR), fontsize=10)
    ax.set_xlabel('cache (width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.violinplot(per_cache_accuracies,
        showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    fig.autofmt_xdate()
    fig.suptitle('        Per-cache test performance')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-violinplot.svg'
        .format(CONFIG_STR)))

    ##### Histogram - accuracy distribution
    fig, ax = pyplot.subplots()
    ax.set_title('{}'
        .format(VISUALS_CONFIG_STR), fontsize=10)
    ax.set_xlabel('Accuracy score')
    ax.set_ylabel('Frequency (log)')
    _, x, _ = ax.hist(all_accuracies, bins=64, density=True, log=True)
    density = stats.gaussian_kde(all_accuracies)
    ax.set_xticks(np.arange(0, 1.1, step=0.1)) # @FIXME ticks with 0.1 steps
    ax.plot(x, density(x))
    fig.suptitle('            Accuracy score distribution')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-histogram.svg'
        .format(CONFIG_STR)))
    
    ##### Boxplot with fold means
    fig, ax = pyplot.subplots()
    ax.set_xlabel('cache (width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.set_title('{}'
        .format(VISUALS_CONFIG_STR), fontsize=10)
    ax.boxplot(per_cache_means, labels=labels)
    fig.autofmt_xdate()
    fig.suptitle('           Per-cache folds mean test performance')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
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
    ax.set_title('{}'
        .format(VISUALS_CONFIG_STR), fontsize=10)
    ax.set_xlabel('Fraction of pixels of type road marking')
    ax.set_ylabel('Accuracy score')

    # Save
    fig.suptitle('         Accuracy score vs. fraction of pixels of type road marking')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-scatterplot.svg'
        .format(CONFIG_STR)))

plot_prediction_img_comparison()
# plot_gt_histogram()
# plot_overall_performance()
# plot_acc_vs_gt_fractions()
