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

print('Grouth truth image shapes:')
for im in gt:
    print(im.shape)
print('Supervised approach image shapes:')
for im in sv:
    print(im.shape)
print('Un-supervised approach image shapes:')
for im in usv:
    print(im.shape)
print('End of program.')

#%%
