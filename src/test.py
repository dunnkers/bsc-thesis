#%%
import pickle
import constants as const
from skimage.io import imread_collection
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import FunctionTransformer

gt  = imread_collection('./cache_100x200/groundtruth/image1.png')
sv  = imread_collection('./cache_100x200/supervised/image1.png')
usv = imread_collection('./cache_100x200/unsupervised/output/output_image1.png')

picklepath = '{}_n={}_SVM.pickle'.format(const.CACHE.path, 1000)
with open(picklepath, 'rb') as handle:
    model = pickle.load(handle)

print('model', model)

# Transform
def im2vec(im):
    """ Flatten 2D image to a 1D vector. Note that `ravel` creates a view,
    possibly distorting the original array. """
    return np.array(im).ravel()

def ic2vecs(ic):
    """ Maps images in collection to a 1D vector. """
    return [im2vec(im) for im in tqdm(ic, desc='Vectorizing')]

# Vectorize
vectorize = FunctionTransformer(ic2vecs, validate=False)
gt_vec  = vectorize.fit_transform(gt)  # replace because using `ravel`
sv_vec  = vectorize.fit_transform(sv)  # replace because using `ravel`
usv_vec = vectorize.fit_transform(usv) # replace because using `ravel`
# Stack
X = np.stack((np.hstack(sv_vec), np.hstack(usv_vec)), axis=-1)

# Predict
print('Predicting')
predictions = model.predict(X)

#%%
acc_score = accuracy_score(gt_vec[0], predictions)
print('acc_score = {}'.format(acc_score))

from matplotlib import pyplot
from skimage.io import imshow
im = np.reshape(predictions, (200, 100))
imshow(im)
pyplot.show()