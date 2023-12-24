import pandas as pd
from os import path, listdir

SAMPLING_RATE = 500

MAIN_PATH = '/Users/danil/AIIJC_FINAL'
DATA_PATH = path.join(MAIN_PATH, 'DATA')
TRAIN_PATH = path.join(DATA_PATH, 'train')
TEST_PATH = path.join(DATA_PATH, 'test')
TRAIN_META_PATH = path.join(TRAIN_PATH, 'train_meta.csv')
TRAIN_GTS_PATH = path.join(TRAIN_PATH, 'train_gts_final.csv')
TEST_META_PATH = path.join(TEST_PATH, 'test_meta.csv')

train_meta = pd.read_csv(TRAIN_META_PATH)
train_gts = pd.read_csv(TRAIN_GTS_PATH)
test_meta = pd.read_csv(TEST_META_PATH)

train_meta = train_meta.merge(train_gts, on=['record_name'])

filenames_train = listdir(TRAIN_PATH)
filenames_test = listdir(TEST_PATH)
filenames_train = list(filter(lambda x: 'npy' in x, filenames_train))
filenames_test = list(filter(lambda x: 'npy' in x, filenames_test))