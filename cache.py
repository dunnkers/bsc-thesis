#%%
import constants as const
from os.path import exists, dirname
from skimage.io import imread_collection, imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.util import img_as_bool, img_as_ubyte
from os import makedirs
from tqdm.auto import tqdm
from warnings import catch_warnings, simplefilter

print('GT_DATA_GLOB   =', const.GT_DATA_GLOB)
print('SV_DATA_GLOB   =', const.SV_DATA_GLOB)
print('USV_DATA_GLOB  =', const.USV_DATA_GLOB)
print('CACHES    =', const.CACHES)

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

        # save
        imsave(path, im, check_contrast=False) # check_contrast is skimage>=0.16

def get_impath_cached(impath, cachepath):
    return impath.replace(const.DATA_PATH, cachepath, 1)

def cache_collection(ic, cache, transform=None, desc='Caching'):
    """
    Checks for every image in this collection whether it is
    cached. In case no, it caches a resized image.

    Parameters:
    ic (ImageCollection): The image collection to write cache for.
    cache (namedtuple): Cache to write to. Namedtuple of (cachepath, shape).
    """
    for idx, impath in enumerate(tqdm(ic.files, desc=desc, unit='imgs')):
        impath_cached = get_impath_cached(impath, cache.path)

        if not exists(impath_cached):
            cache_image(ic[idx], impath_cached, cache.shape, transform=transform)

def cache_all():
    gt  = imread_collection(const.GT_DATA_GLOB)
    sv  = imread_collection(const.SV_DATA_GLOB)
    usv = imread_collection(const.USV_DATA_GLOB)
    for i, cache in enumerate(const.CACHES):
        print('[{}/{}] Writing cache to \'{}\'...'
            .format(i + 1, len(const.CACHES), cache.path))
        cache_collection(gt, cache,  desc='Caching  groundtruth',
            transform=gt_transform)
        cache_collection(sv, cache,  desc='Caching   supervised')
        cache_collection(usv, cache, desc='Caching unsupervised')

def gt_transform(im):
    return img_as_ubyte(img_as_bool(im))

cache_all()
print('Finished caching.')