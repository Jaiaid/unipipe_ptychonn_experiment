import os

SEED = 2661

H,W = 64,64
IMGCOUNT = 161 # How many image
DIFFRLINE = 161 # How many lines per image
NLINES = 5 #How many lines of data to use for training?
NLTEST = 60 #How many lines for the test set?


N_VALID = 805 #How much to reserve for validation

PRETRAIN_EPOCHS = 10
NGPUS = 1
TRAIN_BATCH_SIZE = NGPUS * 32
# is 1 because we are considering continuous data stream
INFERENCE_BATCH_SIZE = 32
LR = NGPUS * 1e-3

# this threshold helps determine when to stop training
LOSS_CHANGE_MIN_THRESHOLD = 1e-3

# taken as mean from QPS range from https://www.mdpi.com/1424-8220/24/16/5262
# initially used 259 got from 20% of whole data as test set served over 10s
# DATARATE_PER_SEC = 259
# # request deadline
# DEADLINE_PER_REQ_SEC = 0.02

EPOCHS = 10

ROOT_DIR = "../../"
DATA_DIFFR_PATH = os.path.join(ROOT_DIR, 'data/20191008_39_diff.npz')
REAL_SPACE_PATH = os.path.join(ROOT_DIR, 'data/20191008_39_amp_pha_10nm_full.npy')
