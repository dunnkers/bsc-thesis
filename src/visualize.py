#%%
from os.path import exists, join

import numpy as np
from joblib import dump, load
from matplotlib import pyplot
from skimage.io import imread, imshow

from constants import CACHES, DUMP_TESTED, OUT_FOLDERNAME, VISUALS_FOLDERPATH


def plot_gt_histogram():
    # pyplot.figure(figsize=(5, 10), dpi=80)
    im = imread('./data/groundtruth/image1.png', as_gray=True)
    pyplot.subplot(1, 2, 1).set_title("Groundtruth image")
    imshow(im)
    ax_hist = pyplot.subplot(1, 2, 2)
    ax_hist.set_title("Histogram in 6-bit bins")
    ax_hist.hist(im.ravel(), bins=64, log=True)
    
    pyplot.tight_layout()
    pyplot.savefig(join(VISUALS_FOLDERPATH, 'groundtruth-histogram.svg'))

def plot_all():
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
        raise Exception("No data to plot")
    
    # Boxplot
    # pyplot.style.use('classic')
    _, ax = pyplot.subplots()
    ax.set_xlabel('(width x height) in pixels')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Results for {}'.format(OUT_FOLDERNAME))
    ax.boxplot(data, labels=labels)
    pyplot.savefig(join(VISUALS_FOLDERPATH, '{}-boxplot.svg'
        .format(OUT_FOLDERNAME)))

plot_gt_histogram()
plot_all()
