#%% [markdown]
# # Road Marker classification
# Fusing two road marker detection algorithms.

#%%
from skimage.io import imread_collection
ic = imread_collection('./data/groundtruth/image?.png', False)
for im in ic:
    print(im.shape)
print('End of program.')