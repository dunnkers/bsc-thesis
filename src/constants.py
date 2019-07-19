from os.path import join
from collections import namedtuple

DATA_PATH = './data'
IMG_GLOB = '*.png'

GT_FOLDERNAME  = 'groundtruth'
SV_FOLDERNAME  = 'supervised'
USV_FOLDERNAME = 'unsupervised/output'
OUT_FOLDERNAME = 'output'

GT_IMAGENAME = 'image'
SV_IMAGENAME = 'image'
USV_IMAGENAME = 'output_image'

GT_DATA_GLOB  = join(DATA_PATH, GT_FOLDERNAME, IMG_GLOB)
SV_DATA_GLOB  = join(DATA_PATH, SV_FOLDERNAME, IMG_GLOB)
USV_DATA_GLOB = join(DATA_PATH, USV_FOLDERNAME, IMG_GLOB)

# Caches to save.
Cache = namedtuple('Cache', ['path', 'shape'])
CACHES = [
    # Cache('./cache_70x140', (140, 70)),
    Cache('./cache_100x200', (200, 100)),
    # Cache('./cache_140x280', (280, 140)),
    # Cache('./cache_175x350', (350, 175)),
    # Cache('./cache_350x700', (700, 350))
]

### Current configuration
# CACHE = CACHES[0] # Cache to use.
MAX_SAMPLES = 100
N_FOLDS = 5
PICKLEFILE_PREPARED = '{}-fold.pickle'.format(N_FOLDS)