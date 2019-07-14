#%%
import constants as const
from os.path import exists, dirname
from skimage.io import imread_collection, imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.util import img_as_bool, img_as_uint
from os import makedirs
from tqdm.auto import tqdm
from warnings import catch_warnings, simplefilter

print('GT_DATA_GLOB   =', const.GT_DATA_GLOB)
print('SV_DATA_GLOB   =', const.SV_DATA_GLOB)
print('USV_DATA_GLOB  =', const.USV_DATA_GLOB)
print('CACHES    =', const.CACHES)

def transform_image(im):
    # ensure grayscale
    im = rgb2gray(im)

    # stretch contrast
    im = rescale_intensity(im)

    return im

def cache_image(im, path, shape, transform=transform_image):
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
        im = transform(im)

        # save
        imsave(path, im, check_contrast=False) # check_contrast is skimage>=0.16

def get_impath_cached(impath, cachepath):
    return impath.replace(const.DATA_PATH, cachepath, 1)

def cache_collection(ic, transform=transform_image, desc='Caching'):
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
                cache_image(ic[idx], impath_cached, shape, transform=transform)

def gt_transform(im):
    im = rgb2gray(im)
    
    return img_as_uint(img_as_bool(im))

    # imsave doesn't accept img_as_bool(non-number values) since skimage=>0.16
    # see https://github.com/imageio/imageio/blob/master/imageio/core/functions.py#L293
    # return img_as_bool(im) # returns more of class; roadmarker=TRUE

images = imread_collection(const.GT_DATA_GLOB)
cache_collection(images, transform=gt_transform, desc='Caching  groundtruth')

images = imread_collection(const.SV_DATA_GLOB)
cache_collection(images, desc='Caching   supervised')

images = imread_collection(const.USV_DATA_GLOB)
cache_collection(images, desc='Caching unsupervised')

print('Finished caching.')