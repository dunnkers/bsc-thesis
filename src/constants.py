from os.path import join
from collections import namedtuple
from numpy import uint8

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
    Cache('./cache_35x70',      (70, 35)),      # 1/20 downscale
    Cache('./cache_50x100',    (100, 50)),    # 1/14 downscale
    # Cache('./cache_70x140',   (140, 70)),     # 1/10 downscale
    Cache('./cache_100x200',  (200, 100)),    # 1/7. downscale
    # Cache('./cache_140x280',  (280, 140)),    # 1/5. downscale
    # Cache('./cache_175x350',  (350, 175)),    # 1/4. downscale
    # Cache('./cache_350x700',  (700, 350))     # 1/2. downscale
]

### Current configuration
MAX_SAMPLES = 100
N_FOLDS = 5
PICKLEFILE_PREPARED = '{}-fold.pickle'.format(N_FOLDS)
CACHE_DATA_TYPE = uint8