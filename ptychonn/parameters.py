import os

SEED = 0

# Iterative Phase Retrieval Throughput
IPR_THROUGHPUT = 16

# for smaller data
H,W = 64,64
IMGCOUNT = 161 # How many image
DIFFRLINE = 161 # How many lines per image
# for larger data
H_L,W_L = 128,128
DIFFRLINE_L = 186 # How many lines per image
SCANPOINT_L = 186 # How many scanpoint per line


NLINES = 5 #How many lines of data to use for training?
NLTEST = 60 #How many lines for the test set?
VALID_PERCENTAGE = 5
TEST_PERCENTAGE = 60 

N_VALID = 805 #How much to reserve for validation

PRETRAIN_EPOCHS = 10
NGPUS = 1
TRAIN_BATCH_SIZE = NGPUS * 64
# is 1 because we are considering continuous data stream
INFERENCE_BATCH_SIZE = 64
LR = NGPUS * 1e-3

# this threshold helps determine when to stop training
LOSS_CHANGE_MIN_THRESHOLD = 1e-3

# taken as mean from QPS range from https://www.mdpi.com/1424-8220/24/16/5262
# initially used 259 got from 20% of whole data as test set served over 10s
# DATARATE_PER_SEC = 259
# # request deadline
# DEADLINE_PER_REQ_SEC = 0.02

EPOCHS = 1000

ROOT_DIR = "../../"
DATA_DIFFR_PATH = os.path.join(ROOT_DIR, 'data/20191008_39_diff.npz')
REAL_SPACE_PATH = os.path.join(ROOT_DIR, 'data/20191008_39_amp_pha_10nm_full.npy')
LARGE_DATASET_DIR = os.path.join(ROOT_DIR, 'data/Tao_tungsten_pattern_data')
LARGE_DATASET_FILE = "data_train_meanSubStdData.h5"
LARGE_DATASET_DIFFRCOUNT = 34596
# LARGE_DATASET_VALID_FILELIST = ["scan205.npz", "scan221.npz", "scan222.npz", "scan223.npz", "scan257.npz"]



SHM_MARKER_TRANSMIT_START = "PTYCHO_STREAM_TRANSMIT_START"
SHM_MARKER_TRANSMIT_END = "PTYCHO_STREAM_TRANSMIT_END"
SHM_MARKER_IPR_INIT_FINISH = "PTYCHO_STREAM_IPR_INIT"

SHM_DATA_DIFFR_NAMEFMT = "{0:06}.raw"
SHM_DATA_GEN_PHASE_NAMEFMT = "{0:06}.rawgtph"
SHM_DATA_GEN_AMP_NAMEFMT = "{0:06}.rawgti"

SHM_MARKER_FMT_GTGENERATION_FOLDER = "PTYCHO_STREAM_GT_IPRINTERVAL_{0:06}"
SHM_MARKER_FMT_IPRINTERVAL_END = "PTYCHO_STREAM_GT_INTERVAL_END_{0:06}"
SHM_MARKER_NNRES_FOLDER = "PTYCHO_STREAM_NN_RESULT"
SHM_MARKER_NNRES_PHASE_NAMEFMT = "{0:06}.rawph"
SHM_MARKER_NNRES_AMP_NAMEFMT = "{0:06}.rawi"