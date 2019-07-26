#%%
from os.path import basename, exists, join

import numpy as np
from joblib import dump, load
from matplotlib import gridspec, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.util import img_as_bool, img_as_ubyte
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from itertools import product

from constants import (CACHES, CONFIG_STR, DATA_PATH, DUMP_TESTED,
                       GT_DATA_GLOB, GT_FOLDERNAME, GT_IMAGENAME, IMG_GLOB,
                       OUT_FOLDERNAME, SV_FOLDERNAME, SV_IMAGENAME,
                       USV_FOLDERNAME, USV_IMAGENAME, VISUALS_CONFIG_STR,
                       VISUALS_FOLDERPATH)


def get_accuracy_map(cachepath):
    """ Returns a 1-D map containing the accuracy scores of this cache, 
        as according to its gt_files array. """
    # Dumpfile path
    cachepath_tested = cachepath.replace('./', './tested/')
    path = join(cachepath_tested, DUMP_TESTED)
    if not exists(path): # skip when cache not tested yet.
        return []

    # Load data from dumpfile
    folded_dataset = load(path)
    gt_files = folded_dataset['gt_files']
    folds = folded_dataset['folds']
    
    # accuracy map
    gt_accs = np.zeros(len(gt_files))

    for fold in folds:
        accuracies = fold['accuracies']
        test_indexes = fold['test_indexes']

        for i in range(len(test_indexes)):
            gt_accs[test_indexes[i]] = accuracies[i]
        
    return (gt_accs, gt_files)

def plot_prediction_img_comparison(cachepath, imagename):
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    imagefile = '{}.png'.format(imagename)
    
    # Image paths
    gtpath  = join(cachepath, GT_FOLDERNAME, imagefile)
    svpath  = gtpath.replace(
            GT_FOLDERNAME, SV_FOLDERNAME).replace(GT_IMAGENAME, SV_IMAGENAME)
    usvpath = gtpath.replace(
            GT_FOLDERNAME, USV_FOLDERNAME).replace(GT_IMAGENAME, USV_IMAGENAME)
    outpath = gtpath.replace(
            # './', './tested/').replace(
                GT_FOLDERNAME, OUT_FOLDERNAME)

    # Skip when not tested yet
    if not exists(outpath):
        return

    # Read images
    gt  = imread(gtpath)
    sv  = imread(svpath)
    usv = imread(usvpath)
    out = imread(outpath)

    ### Find accuracy
    acc_map = get_accuracy_map(cachepath)
    for gt_acc, gt_file in zip(*acc_map):
        if gt_file == gtpath: # file found
            acc = gt_acc
            break

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

    fig.text(0.193, 0.945, '{}, cache={}x{}, im={}'
        .format(VISUALS_CONFIG_STR, w, h, imagename), fontsize=10)

    fig.text(0.54, 0.88, 'contrast stretched', fontsize=10, color='white',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':2})

    fig.text(0.53, 0.45, 'accuracy = {:.4f}'.format(acc), fontsize=11, color='white',
        bbox={'facecolor':'black', 'alpha':0.7, 'pad':2})

    fig.savefig(join(VISUALS_FOLDERPATH, '{}-prediction-{}.svg'
        .format(CONFIG_STR, imagename)), bbox_inches='tight')
    pyplot.close(fig)
    pyplot.clf()


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
    pyplot.close(fig)
    pyplot.clf()

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
    pyplot.close(fig)
    pyplot.clf()

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
    pyplot.close(fig)
    pyplot.clf()
    
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
    pyplot.close(fig)
    pyplot.clf()

def plot_acc_vs_gt_fractions(cachepath):
    gt  = imread_collection(join(cachepath, GT_FOLDERNAME, IMG_GLOB))
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

    # Size
    h, w = gt[0].shape
    
    ####### Accuracies
    gt_accs, _ = get_accuracy_map(cachepath)
    fig, ax = pyplot.subplots()
    ax.scatter(gt_fractions, gt_accs, marker='.', alpha=0.3)
    ax.set_title('{}, cache={}x{}'
        .format(VISUALS_CONFIG_STR, w, h), fontsize=10)
    ax.set_xlabel('Fraction of pixels of type road marking')
    ax.set_ylabel('Accuracy score')

    # Save
    fig.suptitle('         Accuracy score vs. fraction of pixels of type road marking')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-scatterplot.svg'
        .format(CONFIG_STR)))
    pyplot.close(fig)
    pyplot.clf()

def plot_confusion_matrix(cm, w, h, target_names):
    """ Given a sklearn confusion matrix (cm), make a nice plot """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = pyplot.get_cmap('Blues')

    pyplot.figure(figsize=(6, 5))
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.text(0.10, -0.7, 'Confusion matrix', fontsize=12)
    pyplot.title('{}, cache={}x{}'
        .format(VISUALS_CONFIG_STR, w, h), fontsize=10)
    pyplot.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        pyplot.xticks(tick_marks, target_names)
        pyplot.yticks(tick_marks, target_names, rotation=90)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, "{:0.4f}".format(cm_norm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
        pyplot.text(j, i + 0.1, "Pixels: {:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
            .format(accuracy, misclass))
    pyplot.tight_layout()
    # pyplot.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=0.1)
    pyplot.savefig(join(VISUALS_FOLDERPATH, '{}-confusion-matrix.svg'
        .format(CONFIG_STR)))
    pyplot.clf()

def compute_and_plot_confusion_matrix(cachepath):
    print('Computing confusion matrix...')
    gt  = imread_collection(join(cachepath, GT_FOLDERNAME, IMG_GLOB))
    out = imread_collection(join(cachepath, OUT_FOLDERNAME, IMG_GLOB))
    y_true = np.array(gt).ravel()
    y_pred = np.array(out).ravel()
    cm = confusion_matrix(y_true, y_pred).ravel() # cm = tn, fp, fn, tp
    # cm = np.array([[55938320,  3327284],   [294897,  1689499]])
    class_names = ['Non-road marker', 'Road marker']
    print('Confusion matrix computed.')

    # Size
    h, w = gt[0].shape

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cm, w, h, target_names=class_names)
    pyplot.show()





plot_prediction_img_comparison('./cache_175x350', 'image1')
plot_prediction_img_comparison('./cache_175x350', 'image618')
plot_prediction_img_comparison('./cache_175x350', 'image633')
plot_prediction_img_comparison('./cache_175x350', 'image924')

# Low performance
plot_prediction_img_comparison('./cache_175x350', 'image752')
plot_prediction_img_comparison('./cache_175x350', 'image867')
plot_prediction_img_comparison('./cache_175x350', 'image927')
plot_prediction_img_comparison('./cache_175x350', 'image928')

# High performance
plot_prediction_img_comparison('./cache_175x350', 'image633')
plot_prediction_img_comparison('./cache_175x350', 'image623')
plot_prediction_img_comparison('./cache_175x350', 'image632')
plot_prediction_img_comparison('./cache_175x350', 'image849')
plot_prediction_img_comparison('./cache_175x350', 'image299')
plot_prediction_img_comparison('./cache_175x350', 'image846')
plot_prediction_img_comparison('./cache_175x350', 'image628')
plot_prediction_img_comparison('./cache_175x350', 'image291')
plot_prediction_img_comparison('./cache_175x350', 'image284')
plot_prediction_img_comparison('./cache_175x350', 'image290')

plot_gt_histogram()
plot_overall_performance()
plot_acc_vs_gt_fractions('./cache_175x350')
compute_and_plot_confusion_matrix('./cache_175x350')
