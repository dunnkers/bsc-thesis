from os.path import join

DATA_PATH = './data'
IMG_GLOB = '*.png'

GT_DATA_GLOB = join(DATA_PATH, 'groundtruth', IMG_GLOB)
SV_DATA_GLOB = join(DATA_PATH, 'supervised', IMG_GLOB)
USV_DATA_GLOB = join(DATA_PATH, 'unsupervised/output', IMG_GLOB)

# Caches to save.
CACHES = [
    ('./cache_100x200', (200, 100))
]

### Current configuration
CACHE = CACHES[0] # Cache to use.
GT_GLOB = GT_DATA_GLOB.replace(DATA_PATH, CACHE[0], 1)
SV_GLOB = SV_DATA_GLOB.replace(DATA_PATH, CACHE[0], 1)
USV_GLOB = USV_DATA_GLOB.replace(DATA_PATH, CACHE[0], 1)
SAMPLES = 100