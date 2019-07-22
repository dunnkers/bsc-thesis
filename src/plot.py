#%%
from os.path import join, exists

from joblib import dump, load
import numpy as np
from constants import CACHES, DUMP_TESTED, CLASSIFIER
from matplotlib import pyplot

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
    ax.set_title('Results for {}'.format(CLASSIFIER))
    ax.boxplot(data, labels=labels)
    pyplot.show()

# plot_cache(CACHES[0])
plot_all()