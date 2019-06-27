#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection

# Ground truth
gt = imread_collection('./data/groundtruth/image?.png', False)
# Supervised
sv = imread_collection('./data/supervised/image?.png', False)
# Unsupervised
usv = imread_collection('./data/unsupervised/output/output_image?.png', False)

print('gt size: {}, sv size: {}, usv size: {}'.format(
        gt.data.size, sv.data.size, usv.data.size
    ))

#%% [markdown]
# Plot an image of each data type.

#%%
from matplotlib import pyplot
from skimage.io import imshow

pyplot.subplot(2, 3, 1).set_title("Ground truth")
imshow(gt[0])
ax_hist = pyplot.subplot(2, 3, 4)
ax_hist.hist(gt[0].ravel(), bins=256)

pyplot.subplot(2, 3, 2).set_title("Supervised")
imshow(sv[0])
ax_hist = pyplot.subplot(2, 3, 5)
ax_hist.hist(sv[0].ravel(), bins=256)

pyplot.subplot(2, 3, 3).set_title("Unsupervised")
imshow(usv[0])
ax_hist = pyplot.subplot(2, 3, 6)
ax_hist.hist(usv[0].ravel(), bins=256)

pyplot.show()

print('Ground truth:\tshape {}\tmin,max({}, {})'.format(
        gt[0].shape, gt[0].min(), gt[0].max()
    ))
print('Supervised:\tshape {}\tmin,max({}, {})'.format(
        sv[0].shape, sv[0].min(), sv[0].max()
    ))
print('Unsupervised:\tshape {}\tmin,max({}, {})'.format(
        usv[0].shape, usv[0].min(), usv[0].max()
    ))

#%% [markdown]
# Transform image data into feature vectors.

#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class gtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def to_feature_vector(self, img):
        # Use np.vectorize?
        return None

    def to_feature(self, x):
        return None
    
    def transform(self, X, y = None):
        return np.array([
                self.to_feature_vector(img) for img in X
            ])
        # return np.array([) for img in X])

gtTransformer = gtTransformer()
gtTransformer.fit_transform(gt)

# scaler = MinMaxScaler()
# print(scaler.fit(gt))
# print(scaler.data_max_)
# print(scaler.transform(gt))

print('End of program stub.')