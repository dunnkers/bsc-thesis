#%%
import pickle
import constants as const
from sklearn.svm import SVC

picklepath = '{}_n={}.pickle'.format(const.CACHE.path, 1000)
with open(picklepath, 'rb') as handle:
    X, y = pickle.load(handle)

print('X.shape', X.shape)
print('y.shape', y.shape)

modelname = 'SVM'
print('Training {}...'.format(modelname))
model = SVC(gamma = 'auto', verbose=True)
model.fit(X, y)
print('{} trained.'.format(modelname))

picklepath = '{}_n={}_{}.pickle'.format(const.CACHE.path, 1000, modelname)
with open(picklepath, 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
