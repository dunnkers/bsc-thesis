from os.path import join
from collections import namedtuple

DATA_PATH = './data'
IMG_GLOB = '*.png'

GT_FOLDERNAME  = 'groundtruth'
SV_FOLDERNAME  = 'supervised'
USV_FOLDERNAME = 'unsupervised/output'

GT_IMG_GLOB   = join('groundtruth', IMG_GLOB)
GT_DATA_GLOB  = join(DATA_PATH, 'groundtruth', IMG_GLOB)
SV_DATA_GLOB  = join(DATA_PATH, 'supervised', IMG_GLOB)
USV_DATA_GLOB = join(DATA_PATH, 'unsupervised/output', IMG_GLOB)

# Caches to save.
Cache = namedtuple('Cache', ['path', 'shape'])
CACHES = [
    Cache('./cache_100x200', (200, 100)),
    Cache('./cache_140x280', (280, 140))
]
# GT_CACHE_GLOB = lambda cache_path: join(cache_path, 'groundtruth', IMG_GLOB)

### Current configuration
CACHE = CACHES[0] # Cache to use.
GT_GLOB_ALL  = GT_DATA_GLOB.replace(DATA_PATH, CACHE.path, 1)
SV_GLOB_ALL  = SV_DATA_GLOB.replace(DATA_PATH, CACHE.path, 1)
USV_GLOB_ALL = USV_DATA_GLOB.replace(DATA_PATH, CACHE.path, 1)
SAMPLES = 100