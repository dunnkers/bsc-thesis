#%%
import constants as const
print('DATA_PATH =', const.DATA_PATH)
print('GT_GLOB =', const.GT_GLOB)
print('SV_GLOB =', const.SV_GLOB)
print('USV_GLOB =', const.USV_GLOB)

#%% [markdown]
# Setup transformers.

#%%
# from skimage.transform import resize
# from sklearn.preprocessing import FunctionTransformer

# def resize_image(im, shape):
#     resize(im, shape, mode='reflect', anti_aliasing=True)

# resizer = FunctionTransformer(resize_image)

#%%
from os.path import exists, dirname
from skimage.io import imsave
from skimage.transform import resize
# from os.path import splitext, basename, exists
from os import makedirs

# def constructNewPath(path, new_folder, suffix = ''):
#     if not exists(new_folder):
#         makedirs(new_folder)

#     file, ext = splitext(path)
#     return '{}/{}{}{}'.format(
#         new_folder, basename(file), suffix, ext
#     )

def cache_image(im, path, shape):
    """
    Saves a cache version of the image if it does not exist.

    Parameters:
    im ().
    """
    dirpath = dirname(path)
    if not exists(dirpath):
        makedirs(dirpath)
    
    imsave(path, resize(im, shape, mode='reflect', anti_aliasing=True),
        check_contrast=False)
    pass

def get_impath_cached(impath, cachepath):
    return impath.replace(const.DATA_PATH, cachepath, 1)

# def has_cache(impath, cachepath):
#     return exists(impath_cached(impath, cachepath))

def cache_collection(ic):
    """
    Checks for every image in this collection whether it is
    cached. In case no, it caches a resized image.

    Parameters:
    ic (ImageCollection): The image collection to write cache for.
    """
    for idx, impath in enumerate(ic.files):
        for cachepath, shape in const.CACHES:
            impath_cached = get_impath_cached(impath, cachepath)

            if not exists(impath_cached):
                cache_image(ic[idx], impath_cached, shape)

#%%
from skimage.io import imread_collection
images = imread_collection(const.GT_GLOB)
cache_collection(images)

print('end')