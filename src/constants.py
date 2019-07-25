#%%
from os.path import join
from collections import namedtuple
from numpy import uint8

DATA_PATH = './data'
IMG_GLOB = '*.png'

GT_FOLDERNAME  = 'groundtruth'
SV_FOLDERNAME  = 'supervised'
USV_FOLDERNAME = 'unsupervised/output'

GT_IMAGENAME = 'image'
SV_IMAGENAME = 'image'
USV_IMAGENAME = 'output_image'

GT_DATA_GLOB  = join(DATA_PATH, GT_FOLDERNAME, IMG_GLOB)
SV_DATA_GLOB  = join(DATA_PATH, SV_FOLDERNAME, IMG_GLOB)
USV_DATA_GLOB = join(DATA_PATH, USV_FOLDERNAME, IMG_GLOB)

# Caches to save.
Cache = namedtuple('Cache', ['path', 'shape'])
CACHES = [
    Cache('./cache_35x70',      (70, 35)),    # 1/20 downscale
    Cache('./cache_50x100',    (100, 50)),    # 1/14 downscale
    Cache('./cache_70x140',    (140, 70)),    # 1/10 downscale
    Cache('./cache_100x200',  (200, 100)),    # 1/7. downscale
    Cache('./cache_140x280',  (280, 140)),    # 1/5. downscale
    Cache('./cache_175x350',  (350, 175)),    # 1/4. downscale
    Cache('./cache_350x700',  (700, 350)),    # 1/2. downscale
    Cache('./cache_525x1050', (1050, 525))    # 3/4. downscale
]

### Configuration
CACHE_DATA_TYPE = uint8 # 8-bit imaging. So that's; 2^8=256 values, in grayscale.
N_JOBS          = -1    # Parallelization. Use `-1` for using all CPU cores,
                        # or `1` for only using one.
                        # can't start with vscode integrated terminal when `-1`
                        # see: https://github.com/microsoft/ptvsd/issues/943
VERBOSE_LOGGING = False # Log extra messages during training
GT_TRANSFORM    = 'img_as_bool'
                        # Can be either: ('img_as_bool' or 'threshold_yen')

## Variable
MAX_SAMPLES     = 200   # Max. samples per class.
N_FOLDS         = 10    # How many folds for k-fold. The `n_splits` parameter.
CLASSIFIER      = 'SVM'
                        # Can be either: ('SVM' or 'XGBoost').

# Configuration strings
CONFIG_STR_NOCLF = 'max_samples={},folds={}'.format(MAX_SAMPLES, N_FOLDS)
CONFIG_STR       = '{},clf={}'.format(CONFIG_STR_NOCLF, CLASSIFIER)

# Derive filenames off configuration
DUMP_TRANSFORMED = "{}.joblib".format(CONFIG_STR_NOCLF)
DUMP_TRAINED     = "{}.joblib".format(CONFIG_STR)
DUMP_TESTED      = "{},tested.joblib".format(CONFIG_STR)

# Images output folder
OUT_FOLDERNAME   = '{},output'.format(CONFIG_STR)

# Visualization output folder
VISUALS_FOLDERPATH = '../bsc-thesis-report/img/'
VISUALS_CONFIG_STR = 'folds={}, clf={}'.format(N_FOLDS, CLASSIFIER)