#%%
import pickle
import constants as const
from skimage.io import imread_collection
from transform import vectorize
from sklearn.metrics import accuracy_score
import numpy as np

print('GT_GLOB   =', const.GT_GLOB)
print('SV_GLOB   =', const.SV_GLOB)
print('USV_GLOB  =', const.USV_GLOB)
print('CACHE     =', const.CACHE)

gt  = imread_collection('./data/groundtruth/image1.png')
sv  = imread_collection('./data/supervised/image1.png')
usv = imread_collection('./data/unsupervised/output/output_image1.png')

picklepath = '{}_n={}_SVM.pickle'.format(const.CACHE.path, 1000)
with open(picklepath, 'rb') as handle:
    model = pickle.load(handle)

print('model', model)

# Transform
gt_vec  = vectorize.fit_transform(gt)  # replace because using `ravel`
sv_vec  = vectorize.fit_transform(sv)  # replace because using `ravel`
usv_vec = vectorize.fit_transform(usv) # replace because using `ravel`

X = np.stack((np.hstack(sv_vec), np.hstack(usv_vec)), axis=-1)

predictions = model.predict(X)
acc_score = accuracy_score(X, predictions)
print('acc_score = {}'.format(acc_score))