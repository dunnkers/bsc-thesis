#%%
import constants as const
print('DATA_PATH =', const.DATA_PATH)
print('GT_GLOB =', const.GT_GLOB)
print('SV_GLOB =', const.SV_GLOB)
print('USV_GLOB =', const.USV_GLOB)

#%%
from os.path import exists, dirname
from skimage.io import imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from os import makedirs
from tqdm import tqdm

def cache_image(im, path, shape):
    """
    Saves a cache version of the image if it does not exist.

    Parameters:
    im (numpy.ndarray) Image to resize and cache.
    path (string) Path to save cached image to.
    shape (tuple) Shape to resize image to.
    """
    dirpath = dirname(path)
    if not exists(dirpath):
        makedirs(dirpath)
    
    # rescale,.. or,.. 
    # import warnings

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    imsave(path, resize(rescale_intensity(im), shape, mode='reflect', anti_aliasing=True),
        check_contrast=False)
    pass

def get_impath_cached(impath, cachepath):
    return impath.replace(const.DATA_PATH, cachepath, 1)

def cache_collection(ic, desc='Caching image collection'):
    """
    Checks for every image in this collection whether it is
    cached. In case no, it caches a resized image.

    Parameters:
    ic (ImageCollection): The image collection to write cache for.
    """
    for idx, impath in enumerate(tqdm(ic.files, desc=desc)):
        for cachepath, shape in const.CACHES:
            impath_cached = get_impath_cached(impath, cachepath)

            if not exists(impath_cached):
                cache_image(ic[idx], impath_cached, shape)

#%%
from skimage.io import imread_collection

images = imread_collection(const.GT_GLOB)
cache_collection(images, desc='Caching groundtruth')

images = imread_collection(const.SV_GLOB)
cache_collection(images, desc='Caching supervised')

images = imread_collection(const.USV_GLOB)
cache_collection(images, desc='Caching unsupervised')

print('end')