"""
需要的宏变量
"""
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'

DEBUG_MODE = True  # if True, 会打印一些调试信息

UNK_TOKEN = 0
UNK_STR = '__UNK__'

BATCH_SIZE = 128
TRAIN_SIZE = 1000  # utils.preprocess 每次向量化的大小
PAD_SIZE = 500  # 句子词向量的最大长度, 统计分布上约为 80%
EMBED_DIM = 200  # 腾讯词嵌入的维度
HIDDEN_DIM = 300
TRAIN_TEST_SPLIT = 0.9
TRAIN_DATA_REUSE_TIMES = 10  # sampler.sampler 每次 load 的复用次数
MIN_TENCENT_FREQ = 10  # 用于剔除腾讯词嵌入的最低词频

DATA_PATH = '/data/wangchenghao/data/app_joined_%d.pkl'
TRAIN_PATH = '/data/wangchenghao/data/app_train_%d.pkl'
SAVE_PATH = '/data/wangchenghao/baseline_model.zip'
EMBEDDING_PATH = '/data/wangchenghao/tencent_trimmed.pkl'
TRIMMED_EMBEDDING_PATH = '/data/wangchenghao/tencent_trimmed.pkl'
DATA_PATH = '/Users/wangchenghao1103/Desktop/app_demo_%d.pkl'
