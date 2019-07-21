#%%
from datetime import timedelta
from os.path import join
from time import time

from joblib import dump, load
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from constants import (CACHES, CLASSIFIER, DUMP_TRAINED, DUMP_TRANSFORMED,
                       N_JOBS, VERBOSE_LOGGING)


def train_fold(fold):
    X_train, y_train, _, _ = fold['data']

    # Select model
    if CLASSIFIER == 'SVM':
        model = SVC(gamma = 'auto', verbose=VERBOSE_LOGGING)
    elif CLASSIFIER == 'XGBoost':
        raise NotImplementedError('XGBoost not implemented yet.')
    else:
        raise NotImplementedError('`{}` classifier not implemented.'
            .format(CLASSIFIER))

    # Use BaggingClassifier to speed up training
    n_estimators = 10
    clf = BaggingClassifier(model,
        max_samples=1.0 / n_estimators,
        n_estimators=n_estimators,
        n_jobs=N_JOBS)

    # Train.
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print('  {} trained in {:.2f} sec'.format(CLASSIFIER, end - start))

    return clf

def train_cache(cache):
    # Open up prepared cache file
    folded_dataset = load(join(cache.path, DUMP_TRANSFORMED))
    n_splits = folded_dataset['n_splits']
    clfs = []

    # Train every fold
    for i, fold in enumerate(folded_dataset['folds']):
        print(' [{}/{}] Training fold using {} train- and {} test images...'
            .format(i + 1, n_splits, fold['train_indexes'].size,
                                     fold['test_indexes'].size))
        clf = train_fold(fold)
        clfs.append(clf)

    dump(clfs, join(cache.path, DUMP_TRAINED))

def train_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Training cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        train_cache(cache)

start = time()
train_all()
end = time()
print('Finished training in {}'.format(timedelta(seconds=end - start)))
