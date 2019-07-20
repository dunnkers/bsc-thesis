#%%
from datetime import timedelta
from os import makedirs
from os.path import dirname, exists
from time import time
from warnings import catch_warnings, simplefilter

from joblib import Parallel, delayed
from numpy import uint8
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from skimage.io import imread_collection, imsave
from skimage.transform import resize
from skimage.util import img_as_bool, img_as_ubyte, img_as_bool
from tqdm.auto import tqdm

from constants import (CACHE_DATA_TYPE, CACHES, DATA_PATH, GT_DATA_GLOB,
                       N_JOBS, SV_DATA_GLOB, USV_DATA_GLOB)

print('GT_DATA_GLOB   =', GT_DATA_GLOB)
print('SV_DATA_GLOB   =', SV_DATA_GLOB)
print('USV_DATA_GLOB  =', USV_DATA_GLOB)
print('CACHES    =', CACHES)

def cache_image(im, path, shape, transform=None):
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
    
    with catch_warnings(): # prevent contrast warnings.
        simplefilter("ignore")

        # resize & custom transform
        im = resize(im, shape, mode='reflect', anti_aliasing=True)
        im = rgb2gray(im)
        if transform:
            im = transform(im)

        # verify data format
        assert(CACHE_DATA_TYPE == uint8) # only support ubyte convert.
        if not im.dtype == CACHE_DATA_TYPE:
            im = img_as_ubyte(im)

        # save
        imsave(path, im, check_contrast=False) # check_contrast is skimage>=0.16

def cache_collection(ic, cache, transform=None, desc='Caching'):
    """
    Checks for every image in this collection whether it is
    cached. In case no, it caches a resized image.

    Parameters:
    ic (ImageCollection): The image collection to write cache for.
    cache (namedtuple): Cache to write to. Namedtuple of (cachepath, shape).
    """
    cache_impath = lambda impath: impath.replace(DATA_PATH, cache.path, 1)
    should_cache = lambda impath: not exists(impath)

    # Map image filenames to their cache path
    files = map(cache_impath, ic.files)
    files = filter(should_cache, files)
    files = list(files)
    
    # Parallelized caching
    Parallel(n_jobs=N_JOBS)(
        # wrap cache_image() with `delayed`
        delayed(cache_image)(ic[idx], impath, cache.shape, transform=transform)
        for idx, impath in enumerate(tqdm(files, desc=desc, unit='imgs'))
    )

def cache_all():
    gt  = imread_collection(GT_DATA_GLOB)
    sv  = imread_collection(SV_DATA_GLOB)
    usv = imread_collection(USV_DATA_GLOB)

    for i, cache in enumerate(CACHES):
        print('[{}/{}] Writing cache to \'{}\'...'
            .format(i + 1, len(CACHES), cache.path))
        cache_collection(gt, cache,  desc='Caching  groundtruth',
            transform=gt_transform)
        cache_collection(sv, cache,  desc='Caching   supervised')
        cache_collection(usv, cache, desc='Caching unsupervised')

def gt_transform(im):
    return img_as_ubyte(img_as_bool(im))
    # return img_as_ubyte(im > threshold_yen(im))

start = time()
cache_all()
end = time()
print('Finished caching in {}'.format(timedelta(seconds=end - start)))
