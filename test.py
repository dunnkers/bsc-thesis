#%%
import pickle
import constants as const

picklepath = '{}_n={}_SVM.pickle'.format(const.CACHE.path, 1000)
with open(picklepath, 'rb') as handle:
    model = pickle.load(handle)

print('model', model)
