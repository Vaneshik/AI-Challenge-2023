import pandas as pd
import tqdm
from constants import train_meta

from constants import test_meta, TEST_PATH
from os import path, listdir
from tqdm.auto import tqdm
from functions import clean_signal, read_signal, dump_pickle

import warnings
warnings.filterwarnings("ignore")


def get_dataset(filepath):
	signals, names, r_names = [], [], []
	for name in tqdm(listdir(filepath)):
		signal = read_signal(path.join(filepath, name))
		signal = clean_signal(signal)
		signals.append(signal)
		names.append(name)
		r_names.append(name[:8])
	return pd.DataFrame({
		'signal': signals,
		'record_name': r_names,
		'signal_name': names,
	})
		
	
df = get_dataset('/Users/danil/AIIJC_FINAL/DATASETS/augmentation/train_augmentation')
dump_pickle('/Users/danil/AIIJC_FINAL/DATASETS/train_aug_full.pickle', df)