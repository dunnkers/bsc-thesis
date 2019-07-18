#%%
import pickle
from os.path import join
from time import time

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from constants import CACHES, PICKLEFILE_PREPARED

def train_fold(fold):
    X_train, y_train, X_test, y_test = fold
    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)

    modelname = 'SVM'
    start = time()
    print('Training {}...'.format(modelname))
    model = SVC(gamma = 'auto', verbose=True)
    n_estimators = 10
    clf = BaggingClassifier(model,
        max_samples=1.0 / n_estimators,
        n_estimators=n_estimators,
        n_jobs=-1) # can't start using vscode terminal; https://github.com/microsoft/ptvsd/issues/943
    clf.fit(X_train, y_train)
    end = time()
    print('{} trained in {}.'.format(modelname, end - start))

    # picklepath = '{}_n={}_{}.pickle'.format(cache.path, 1000, modelname)
    # with open(picklepath, 'wb') as handle:
    #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_cache(cache):
    # Open up prepared cache file
    picklepath = join(cache.path, PICKLEFILE_PREPARED)
    with open(picklepath, 'rb') as handle:
        folded_dataset = pickle.load(handle)
        n_splits = folded_dataset['n_splits']

        for i, fold in enumerate(folded_dataset['folds']):
            print('[{}/{}] Training fold...'.format(i + 1, n_splits))
            train_fold(fold)

print('Training cache 0...')
train_cache(CACHES[0])
print('Finished training.')
