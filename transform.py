#%%
import constants as const
from skimage.io import imread_collection

print('GT_GLOB   =', const.GT_GLOB)
print('SV_GLOB   =', const.SV_GLOB)
print('USV_GLOB  =', const.USV_GLOB)
print('CACHE     =', const.CACHE)

gt = imread_collection(const.GT_GLOB)
sv = imread_collection(const.SV_GLOB)
usv = imread_collection(const.USV_GLOB)

print('end')
