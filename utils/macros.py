"""

"""
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'

DEBUG_MODE = True

UNK_TOKEN = 0
UNK_STR = '__UNK__'

MAX_SENTENCE_LEN = 128
BATCH_SIZE = 16
TRAIN_SIZE = 10000
PAD_SIZE = 10000
TRAIN_TEST_SPLIT = 0.9

VOCABULARY_PATH = '../data/vocabulary.txt'
DATA_PATH = '/data/wangchenghao/data/app_joined_%d.pkl'
TRAIN_PATH = '/data/wangchenghao/data/app_train_%d.pkl'
SAVE_PATH = '/data/wangchenghao/baseline_model.zip'

