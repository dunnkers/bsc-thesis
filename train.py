#%%
import pickle
import constants as const
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import time
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

picklepath = '{}_n={}.pickle'.format(const.CACHE.path, 1000)
with open(picklepath, 'rb') as handle:
    X, y = pickle.load(handle)

# X=X[:100000]
# y=y[:100000]
print('X.shape', X.shape)
print('y.shape', y.shape)

modelname = 'SVM'
start = time.time()
print('Training {}...'.format(modelname))
model = SVC(gamma = 'auto', verbose=True)
n_estimators = 10
clf = BaggingClassifier(model,
    max_samples=1.0 / n_estimators,
    n_estimators=n_estimators,
    n_jobs=-1) # can't start using vscode terminal; https://github.com/microsoft/ptvsd/issues/943
clf.fit(X, y)
end = time.time()
print('{} trained in {}.'.format(modelname, end - start))

picklepath = '{}_n={}_{}.pickle'.format(const.CACHE.path, 1000, modelname)
with open(picklepath, 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
