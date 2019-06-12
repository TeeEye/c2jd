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
TRAIN_SIZE = 1000
PAD_SIZE = 500
EMBED_DIM = 200
TRAIN_TEST_SPLIT = 0.9
TRAIN_DATA_REUSE_TIMES = 10
MAX_SENTENCE_LEN = 1000

DATA_PATH = '/data/wangchenghao/data/app_joined_%d.pkl'
TRAIN_PATH = '/data/wangchenghao/data/app_train_%d.pkl'
SAVE_PATH = '/data/wangchenghao/baseline_model.zip'
EMBEDDING_PATH = '/data/wangchenghao/tencent_embedding.pkl'
# TRAIN_PATH = '/Users/wangchenghao1103/Desktop/app_demo_%d.pkl'
# SAVE_PATH = '/Users/wangchenghao1103/Desktop/baseline_model.zip'
