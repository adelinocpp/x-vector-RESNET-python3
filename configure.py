# Wave path
# TRAIN_WAV_DIR = '/home/admin/Desktop/read_25h_2/train'
TRAIN_WAV_DIR = '../../00_Bases_de_Dados/WhatsApp_01_8k_wav'
# DEV_WAV_DIR = '/home/admin/Desktop/read_25h_2/dev'
TEST_WAV_DIR = '../../00_Bases_de_Dados/CEFALA_8k_Refinados'

# Feature path
TRAIN_FEAT_DIR = 'feat_logfbank_nfilt40/train'
TEST_FEAT_DIR = 'feat_logfbank_nfilt40/test'

# Vector path
TRAIN_VECTOR_DIR = 'vectors_from_training/'
TEST_VECTOR_DIR = 'vectors_from_testing/'
LDA_SAVE_MODELS_DIR = 'LDA_saved/'
LDA_FILE = 'lda_model.txt'
SPHERING_FILE = 'sphering_model.txt'
PLDA_FILE = 'plda_model.txt'
CALIBRATE_MTX_FILE = 'afinity_matrix.txt'
CALIBRATE_THR_FILE = 'threshoold.txt'

TEST_RESULTS_DIR = 'test_results/'
TEST_CONF_MTX = 'conf_mtx_test.txt'

UBM_FILE_NAME = 'ubm_data_file.p'
SAVE_MODELS_DIR = 'model_saved'
# Context window size
NUM_WIN_SIZE = 50 #10 # 100

# RESNET  settings
BACKBONETYPE = 'resnet152'

# LNORM
ALPHA_LNORM = 10
# Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = True
USE_SCALE = False
SAMPLE_RATE = 8000
FILTER_BANK = 40


