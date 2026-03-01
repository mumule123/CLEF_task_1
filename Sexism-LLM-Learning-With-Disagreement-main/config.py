import os
#Hyperparameters
EPOCHS = 10 #10
MAX_LEN = 100
DROPOUT = 0.3
LR = 3e-5 #5e-6, 1e-5, 3e-5, 5e-5
BATCH_SIZE = 20 #24
TRANSFORMERS = ['bert-base-multilingual-uncased','xlm-roberta-base']

N_ROWS=None #16
SEED = 17
CODE_PATH = os.path.abspath(os.path.dirname(__file__))
print(CODE_PATH)
DATA_PATH = CODE_PATH + '/' 
PACKAGE_PATH = CODE_PATH + '/' + 'package'
DATA_DEV_PATH = PACKAGE_PATH + '/' + 'dev/EXIST2025_dev' + '.json'
DATA_TRAIN_PATH = PACKAGE_PATH + '/' + 'training/EXIST2025_training' + '.json'
DATA_TEST_PATH = PACKAGE_PATH + '/' + 'test/EXIST2025_test_clean' + '.json'
LABEL_GOLD_PATH = CODE_PATH + '/' + 'golds'

# 添加本地模型路径
MODEL_PATH = CODE_PATH + '/model'
REPO_PATH = MODEL_PATH  # 设置REPO_PATH为本地模型路径

DATA = 'EXIST2025'
DATA_URL = 'URL_SHARED_BY_ORGANIZERS_TO_DOWNLOAD_EXIST_2025_DATA'

# 修改TRANSFORMERS列表，使用本地模型目录名
TRANSFORMERS = ['xlm-roberta-large']

LABELS = ['task1']
COLUMN_TEXT = 'tweet'
COLUMN_LABELS = 'soft_label_'
DATASET_INDEX = 'id_EXIST'
UNITS = {'task1': 2}

DATASET_TRAIN = 'EXIST2025_training.csv'
DATASET_DEV = 'EXIST2025_dev.csv'
DATASET_TEST = 'EXIST2025_test.csv'
DATASET_TRAIN_DEV = 'EXIST2025_training-dev.csv'

# GPU设置
DEVICE = 'max'  # None 'max' 'cpu' 'cuda:0' 'cuda:1'
GPU_IDS = [0, 2]  # 指定要使用的GPU ID列表
TRAIN_WORKERS = 10
VAL_WORKERS = 10
LOGS_PATH = CODE_PATH + '/' + 'logs'
