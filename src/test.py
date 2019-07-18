#%%
import pickle
from os.path import join

import numpy as np
from matplotlib import pyplot
from skimage.io import imread_collection, imsave, imshow
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm

from constants import CACHES, PICKLEFILE_PREPARED


def test_fold(fold):
    _, _, X_test, y_test = fold['data']
    clf = fold['clf']
    size = fold['size']
    # cachepath = #fold['cachepath']
    shape = (200, 100) #fold['shape']
    
    # Predict
    print('Predicting')
    X_test_split = np.split(X_test, size) # only test one image for now.
    predictions = clf.predict(X_test_split[0]) # ⚠️ Only start using terminal.

    # Accuracy score
    print('Computing accuracy')
    y_test_split = np.split(y_test, size)
    acc_score = accuracy_score(y_test_split[0], predictions)
    print('acc_score = {}'.format(acc_score))

    # Reconstruction
    im = np.reshape(predictions, shape)
    # imsave('duder.png', im, check_contrast=False)
    imshow(im)
    pyplot.show()

def test_cache(cache):
    picklepath = join(cache.path, PICKLEFILE_PREPARED)
    with open(picklepath, 'rb') as handle:
        folded_dataset = pickle.load(handle)
        n_splits = folded_dataset['n_splits']
        
        # Test every fold
        for i, fold in enumerate(folded_dataset['folds']):
            print('[{}/{}] Testing fold...'.format(i + 1, n_splits))
            test_fold(fold)


test_cache(CACHES[0])
print('Finished testing.')
