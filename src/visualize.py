#%%
from itertools import product
from os.path import basename, exists, join

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats
from skimage.color import rgb2gray
from skimage.io import imread, imread_collection, imshow
from skimage.util import img_as_bool, img_as_ubyte
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from constants import (CACHES, CONFIG_STR, CONFIG_STR_NOCLF, DATA_PATH,
                       DUMP_TESTED, FEATURE_VECTOR, GT_DATA_GLOB,
                       GT_FOLDERNAME, GT_IMAGENAME, IMG_GLOB, N_FOLDS,
                       OUT_FOLDERNAME, SV_FOLDERNAME, SV_IMAGENAME,
                       USV_FOLDERNAME, USV_IMAGENAME, VISUALS_CONFIG_STR,
                       VISUALS_FOLDERPATH)


def getConfigStr(clf):
    """ Produce exactly the same config str as in constants, but then in a
        method such that a custom classifier can be used. """
    configstr = '{},clf={}'.format(CONFIG_STR_NOCLF, clf)
    outfolder = '{},output'.format(configstr)
    visualstr = 'folds={}, clf={}'.format(N_FOLDS, clf)
    dumppath  = '{},tested.joblib'.format(configstr)

    return (configstr, outfolder, visualstr, dumppath)

def get_accuracy_map(cachepath):
    """ Returns a 1-D map containing the accuracy scores of this cache, 
        as according to its gt_files array. """
    # Dumpfile path
    cachepath_tested = cachepath.replace('./', './tested/')
    path = join(cachepath_tested, DUMP_TESTED)
    if not exists(path): # skip when cache not tested yet.
        return ([], [])

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

def plot_prediction_img_comparison(cachepath, imagename, clf='XGBoost'):
    """ Plot comparison chart between groundtruth, supervised, unsupervised-
        and the prediction. """
    imagefile = '{}.png'.format(imagename)
    configstr, outfolder, visualstr, _ = getConfigStr(clf)

    # Image paths
    gtpath  = join(cachepath, GT_FOLDERNAME, imagefile)
    svpath  = gtpath.replace(
            GT_FOLDERNAME, SV_FOLDERNAME).replace(GT_IMAGENAME, SV_IMAGENAME)
    usvpath = gtpath.replace(
            GT_FOLDERNAME, USV_FOLDERNAME).replace(GT_IMAGENAME, USV_IMAGENAME)
    outpath = gtpath.replace(
            # './', './tested/').replace(
                GT_FOLDERNAME, outfolder)

    # Skip when not tested yet
    if not exists(outpath):
        return

    # Read images
    gt  = imread(gtpath)
    sv  = imread(svpath)
    usv = imread(usvpath)
    out = imread(outpath)

    ### Find accuracy
    acc = None
    acc_map = get_accuracy_map(cachepath)
    for gt_acc, gt_file in zip(*acc_map):
        if gt_file == gtpath: # file found
            acc = gt_acc
            break

    if acc == None:
        print('Accuracy not found for', imagename)
        return

    # Size
    h, w = gt.shape

    # > sidenote: for images of smaller height, set aspect=False,
    # along with a `aspect='auto'` for each imshow() call.

    # Plot 2x2
    fig = plt.figure(figsize=(7.0, 9.0))
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
        .format(visualstr, w, h, imagename), fontsize=10)

    fig.text(0.54, 0.88, 'contrast stretched', fontsize=10, color='white',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':2})

    fig.text(0.53, 0.45, 'accuracy = {:.4f}'.format(acc), fontsize=11, color='white',
        bbox={'facecolor':'black', 'alpha':0.7, 'pad':2})

    fig.savefig(join(VISUALS_FOLDERPATH, '{}-prediction-{}.svg'
        .format(configstr, imagename)), bbox_inches='tight')
    plt.close(fig)
    plt.clf()


def plot_gt_histogram():
    """ Plot groundtruth image and its associated histogram. """
    # Read
    path = join(DATA_PATH, GT_FOLDERNAME, GT_IMAGENAME + '1.png')
    im = imread(path, as_gray=True)
    
    # 2-col plot
    fig, _ = plt.subplots(1, 2)

    # Image
    plt.subplot(1, 2, 1).set_title("Groundtruth image")
    imshow(im)

    # Histogram
    ax_hist = plt.subplot(1, 2, 2)
    ax_hist.set_title("Histogram in 6-bit bins")
    ax_hist.hist(im.ravel(), bins=64, log=True)
    
    # Save
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, 'groundtruth-histogram.svg'))
    plt.close(fig)
    plt.clf()

def compare_classifiers_performance():
    comparison_boxplot = []

    for cache in CACHES:
        cachepath = cache.path.replace('./', './tested/')
        h, w = cache.shape
        cachelabel = '{}x{}'.format(w, h)

        clfs = ['SVM', 'XGBoost']
        for clf in clfs:
            dump_tested = '{},clf={},tested.joblib'.format(
                    CONFIG_STR_NOCLF, clf)
            path = join(cachepath, dump_tested)
            if exists(path): # skip when cache not tested yet.
                # Load data from dumpfile
                folded_dataset = load(path)
                folds = folded_dataset['folds']
                
                for fold in folds:
                    fold_accuracies = fold['accuracies']
                    fold_acc = np.mean(fold_accuracies)
                    comparison_boxplot.append(dict(
                        accuracy=fold_acc,
                        Classifier=clf,
                        cache=cachelabel
                    ))

    # No data
    if len(comparison_boxplot) == 0:
        print('No data for plotting. ', 'compare_classifiers_performance()')
        return

    ##### Classifier comparison boxplots
    sns.set(style="whitegrid")
    df = pd.DataFrame(comparison_boxplot)
    fig, (ax) = plt.subplots(1, 1)
    ax = sns.boxplot(x="cache", y="accuracy", hue="Classifier",
        data=df)
    ax.set_title('{}-fold cache/classifier performance'.format(N_FOLDS))
    plt.xticks(rotation=30)
    ax.set(xlabel='Cache (width x height) in pixels',
           ylabel='Accuracy score')
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-classifiers-cache-performance.svg'
        .format(CONFIG_STR_NOCLF)))
    plt.close(fig)
    plt.clf()

    ##### Cache comparison - classifiers aggregated.
    sns.set(style="whitegrid")
    fig, (ax) = plt.subplots(1, 1)

    # https://stackoverflow.com/questions/21912634/how-can-i-sort-a-boxplot-in-pandas-by-the-median-values
    grouped = df.groupby(["cache"])
    df2 = pd.DataFrame({col:vals['accuracy'] for col,vals in grouped})
    meds = df2.median()
    meds.sort_values(ascending=True, inplace=True)
    df2 = df2[meds.index]

    ax = sns.boxplot(data=df2,palette="Set3")
    ax.set_title('{}-fold cache performance'.format(N_FOLDS))
    plt.xticks(rotation=30)
    ax.set(xlabel='Cache (width x height) in pixels',
           ylabel='Accuracy score')
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-cache-performance.svg'
        .format(CONFIG_STR_NOCLF)))
    plt.close(fig)
    plt.clf()

    ##### Classifiers comparison - caches aggregated.
    sns.set(style="whitegrid")
    fig, (ax) = plt.subplots(1, 1)

    # https://stackoverflow.com/questions/21912634/how-can-i-sort-a-boxplot-in-pandas-by-the-median-values
    grouped = df.groupby(["Classifier"])
    df2 = pd.DataFrame({col:vals['accuracy'] for col,vals in grouped})
    meds = df2.median()
    meds.sort_values(ascending=True, inplace=True)
    df2 = df2[meds.index]

    ax = sns.boxenplot(data = df2)
    ax.set_title('{}-fold classifier performance'.format(N_FOLDS))
    plt.xticks(rotation=30)
    ax.set(xlabel='Classifier',
           ylabel='Accuracy score')
    fig.tight_layout()
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-classifiers-performance.svg'
        .format(CONFIG_STR_NOCLF)))
    plt.close(fig)
    plt.clf()

def plot_overall_performance(clf='XGBoost'):
    """ Compare cache performance by plotting several boxplots, resembling
        mean fold accuracies. """
    configstr, _, visualstr, dumppath = getConfigStr(clf)

    labels = []
    per_cache_accuracies = []
    all_accuracies = []

    for cache in CACHES:
        cachepath = cache.path.replace('./', './tested/')
        path = join(cachepath, dumppath)
        if not exists(path): # skip when cache not tested yet.
            continue

        # Load data from dumpfile
        folded_dataset = load(path)
        folds = folded_dataset['folds']
        
        # accuracy distribution
        cache_accuracies = []
        for fold in folds:
            fold_accuracies = fold['accuracies']
            all_accuracies.extend(fold_accuracies)
            cache_accuracies.extend(fold_accuracies)
        per_cache_accuracies.append(cache_accuracies)

        # Attach pixel configuration label
        h, w = cache.shape
        labels.append('{}x{}'.format(w, h))
    
    if len(all_accuracies) == 0: # Nothing to plot
        print('No boxplot plotted! - no data for current config found!',
            'plot_overall_performance()')
        return

    ##### Violin plot - accuracy performance & distribution
    fig, ax = plt.subplots()
    ax.set_title('{}'
        .format(visualstr), fontsize=10)
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
        .format(configstr)))
    plt.close(fig)
    plt.clf()

    ##### Histogram - accuracy distribution
    fig, ax = plt.subplots()
    ax.set_title('{}'
        .format(visualstr), fontsize=10)
    ax.set_xlabel('Accuracy score')
    ax.set_ylabel('Frequency (log)')
    _, x, _ = ax.hist(all_accuracies, bins=64, density=True, log=True)
    density = stats.gaussian_kde(all_accuracies)
    ax.set_xticks(np.arange(0, 1.1, step=0.1)) # @FIXME ticks with 0.1 steps
    ax.plot(x, density(x))
    fig.suptitle('            Accuracy score distribution')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=1.0)
    fig.savefig(join(VISUALS_FOLDERPATH, '{}-histogram.svg'
        .format(configstr)))
    plt.close(fig)
    plt.clf()

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
    if len(gt_accs) == 0:
        print('No gt_accs found')
        return
    assert(len(gt_fractions) == len(gt_accs))

    fig, ax = plt.subplots()
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
    plt.close(fig)
    plt.clf()

def plot_confusion_matrix(cm, w, h, new_configstr, target_names):
    """ Given a sklearn confusion matrix (cm), make a nice plot """
    configstr, _, visualstr, _ = new_configstr

    # Compute acc/misclass
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.text(0.10, -0.7, 'Confusion matrix', fontsize=12)
    plt.title('{}, cache={}x{}'
        .format(visualstr, w, h), fontsize=10)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names, rotation=90)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm_norm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
        plt.text(j, i + 0.1, "Pixels: {:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
            .format(accuracy, misclass))
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.92], pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig(join(VISUALS_FOLDERPATH, '{}-confusion-matrix.svg'
        .format(configstr)))
    plt.clf()

def compute_and_plot_confusion_matrix(cachepath, clf='XGBoost'):
    new_configstr = getConfigStr(clf)
    outfolder = new_configstr[1]

    gt  = imread_collection(join(cachepath, GT_FOLDERNAME, IMG_GLOB))
    out = imread_collection(join(cachepath, outfolder,     IMG_GLOB))
    assert(len(gt.files) == len(out.files))

    # Compute cm in incremental fashion
    cm = np.zeros((2, 2))
    for i in tqdm(range(len(gt.files)), desc="Computing confusion matrix"):
        y_true = gt[i].ravel()
        y_pred = out[i].ravel()
        cm = cm + confusion_matrix(y_true, y_pred)

    # Size
    h, w = gt[0].shape

    # Plot confusion matrix
    class_names = ['Non-road marker', 'Road marker']
    plot_confusion_matrix(cm, w, h, new_configstr, target_names=class_names)

def plot_pipeline_performance():
    # 10-FOLD USING 1000 TRAIN/TEST IMAGES
    # Intel® Core™ i7-3820 Quad CPU @ 3.60GHz × 8, 64-bit Ubuntu 19.09, 7,7 GiB RAM

    # transform = imgs per second during transformation sampling.
    # train = train times in seconds per fold
    # test = test time in images per second
    df = pd.DataFrame([
            dict(cache='35x70', transform=3286.1369999999997),
            dict(cache='50x100', transform=3009.1879999999996),
            dict(cache='70x140', transform=2591.011),
            dict(cache='100x200', transform=2040.569),
            dict(cache='140x280', transform=1414.0760000000002),
            dict(cache='175x350', transform=1072.813),
            dict(cache='350x700', transform=326.947),
            dict(cache='525x1050', transform=146.06),
            dict(cache='35x70', Classifier='SVM', test=0.5847953216374269, train=14.907499999999999),
            dict(cache='50x100', Classifier='SVM', test=1.3500742540839745, train=17.1125),
            dict(cache='70x140', Classifier='SVM', test=2.7203482045701852, train=16.884999999999998),
            dict(cache='100x200', Classifier='SVM', test=6.030017426750363, train=17.04),
            dict(cache='140x280', Classifier='SVM', test=11.730067682490528, train=17.5975),
            dict(cache='175x350', Classifier='SVM', test=18.830015798383254, train=18.0075),
            dict(cache='350x700', Classifier='SVM', test=77.41136398823348, train=18.465999999999998),
            dict(cache='525x1050', Classifier='SVM', test=157.83000162091412, train=17.682),
            dict(cache='35x70', Classifier='XGBoost', test=0.02391200382592061, train=10.471),
            dict(cache='50x100', Classifier='XGBoost', test=0.03562522265764161, train=10.075999999999999),
            dict(cache='70x140', Classifier='XGBoost', test=0.060753341433778855, train=10.18),
            dict(cache='100x200', Classifier='XGBoost', test=0.10626992561105207, train=10.194999999999999),
            dict(cache='140x280', Classifier='XGBoost', test=0.2008032128514056, train=10.209),
            dict(cache='175x350', Classifier='XGBoost', test=0.35587188612099646, train=10.213999999999999),
            dict(cache='350x700', Classifier='XGBoost', test=1.1799410029498525, train=10.360000000000003),
            dict(cache='525x1050', Classifier='XGBoost', test=2.9866794098321487, train=10.245999999999999),
        ])

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    sns.set(style="whitegrid")

    ### (1) Transformation speed
    sns.barplot(x="cache", y="transform",
        data=df, palette="Set3", ax=ax1)
    ax1.set_title('Transformations performance, {}-fold'.format(N_FOLDS))
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set(ylabel='imgs sampled / sec', xlabel='')

    ### (2) Train speed
    sns.barplot(x="cache", y="train", hue="Classifier",
        data=df, ax=ax2)
    ax2.set_title('Train performance, {}-fold'.format(N_FOLDS))
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set(ylabel='imgs trained / sec', xlabel='')

    ### (3) Test speed
    sns.barplot(x="cache", y="test", hue="Classifier",
        data=df, ax=ax3)
    ax3.set_title('Test performance, {}-fold'.format(N_FOLDS))
    ax3.axhline(0, color="k", clip_on=False)
    ax3.set(xlabel='Cache (width x height) in pixels',
           ylabel='imgs tested / sec', yscale='log')
    
    # Finalize
    plt.xticks(rotation=30)
    sns.despine(bottom=True)
    f.tight_layout(h_pad=2)
    f.savefig(join(VISUALS_FOLDERPATH, '{}-pipeline-performance.svg'
        .format(CONFIG_STR_NOCLF)))
    plt.close(f)
    plt.clf()

# Low accuracy score - black gt image
plot_prediction_img_comparison('./cache_175x350', 'image752')

# Average accuracy - interesting road condition
plot_prediction_img_comparison('./cache_175x350', 'image631')

### Appendix. Interesting road markings.
plot_prediction_img_comparison('./cache_175x350', 'image633') # high acc
plot_prediction_img_comparison('./cache_175x350', 'image12')
plot_prediction_img_comparison('./cache_175x350', 'image359')
plot_prediction_img_comparison('./cache_175x350', 'image738')
plot_prediction_img_comparison('./cache_175x350', 'image853')
## Appendix - bad ground truth image?
plot_prediction_img_comparison('./cache_175x350', 'image924')
## Appendix - shadows?
plot_prediction_img_comparison('./cache_175x350', 'image639', clf='XGBoost')
plot_prediction_img_comparison('./cache_175x350', 'image639', clf='SVM')

plot_gt_histogram()
plot_overall_performance()
plot_acc_vs_gt_fractions('./cache_175x350')
compute_and_plot_confusion_matrix('./cache_350x700', clf='SVM')
compute_and_plot_confusion_matrix('./cache_350x700', clf='XGBoost')
compare_classifiers_performance()
plot_pipeline_performance()
