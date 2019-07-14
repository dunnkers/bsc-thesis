from os.path import join

DATA_PATH = './data'
IMG_GLOB = '*.png'

GT_GLOB = join(DATA_PATH, 'groundtruth', IMG_GLOB)
SV_GLOB = join(DATA_PATH, 'supervised', IMG_GLOB)
USV_GLOB = join(DATA_PATH, 'unsupervised/output', IMG_GLOB)


# CACHE_PATH = './cache_'
# """ Will save cached versions of all these sizes. """
# CACHE_SHAPES = [
#     (200, 100)
# ]

# Caches to save.
CACHES = [
    ('./cache_100x200', (200, 100))
]