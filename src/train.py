#%%
import pickle
from os.path import join
from time import time

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from constants import CACHES, PICKLEFILE_PREPARED


def train_fold(fold):
    X_train, y_train, X_test, y_test = fold['data']
    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)

    # Select model
    start = time()
    modelname = 'SVM'
    print('Training {}...'.format(modelname))
    model = SVC(gamma = 'auto', verbose=True)

    # Train. Use BaggingClassifier to speed up training
    n_estimators = 10
    clf = BaggingClassifier(model,
        max_samples=1.0 / n_estimators,
        n_estimators=n_estimators,
        # can't start using vscode terminal; https://github.com/microsoft/ptvsd/issues/943
        n_jobs=-1)
    clf.fit(X_train, y_train)

    # Print time
    end = time()
    print('{} trained in {}.'.format(modelname, end - start))

    # Store classifier in fold.
    fold['clf'] = clf

    return fold

def train_cache(cache):
    # Open up prepared cache file
    picklepath = join(cache.path, PICKLEFILE_PREPARED)
    with open(picklepath, 'rb') as handle:
        folded_dataset = pickle.load(handle)
        n_splits = folded_dataset['n_splits']
        folds = []

        # Train every fold
        for i, fold in enumerate(folded_dataset['folds']):
            print('[{}/{}] Training fold...'.format(i + 1, n_splits))
            trained_fold = train_fold(fold)
            folds.append(trained_fold)

        # Overwrite folds attribute in dict with `clf` attached to every fold.
        folded_dataset['folds'] = folds

        with open(picklepath, 'wb') as handle:
            pickle.dump(folded_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_all():
    for i, cache in enumerate(CACHES):
        print('[{}/{}] Training cache \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        train_cache(cache)

train_all()
print('Finished training.')
