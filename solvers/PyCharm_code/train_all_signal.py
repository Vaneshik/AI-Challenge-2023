import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.optim import AdamW, Adam
import os
import shutil

from functions import read_pickle
from MODELS.FullSignal.full_conv1d_front import Model1DPered
from MODELS.FullSignal.full_conv1d_pereg import Model1DPereg
from MODELS.FullSignal.full_conv1d_frontpereg import Model1DPeredPereg
from MODELS.FullSignal.full_conv1d_down import Model1DDown
from MODELS.FullSignal.full_conv1d_norm import Model1DNorm
from MODELS.FullSignal.full_conv1d_frontside import Model1DSideDown
from MODELS.FullSignal.full_conv1d_multilabel import Model1DMultilabel
from MODELS.FullSignal.se_resnet import Se_Resnet
from MODELS.FullSignal.conv_bilstm import CNN
from MODELS.FullSignal.rnn_model import RNNModel
from MODELS.FullSignal.norm_conv import CNNLstm
from constants import train_meta, train_gts
from trainer import Trainer


def moving_avg(x, n):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[n:] - cumsum[:-n]) / float(n)


def smoothing(signal: np.ndarray) -> np.ndarray:
	for i in range(2, 5):
		signal = moving_avg(signal, i)
	return signal


class Dataset1D(Dataset):
	def __init__(self, df: pd.DataFrame, target=None, is_train=True, indexes: list[int]=[0, 1, 2, 3]):
		self.df = df.copy()
		self.target = target
		self.is_train = is_train
		# self.df['signal'] = self.df['signal'].apply(lambda x: scipy.signal.medfilt(x, 3))
		# self.df['signal'] = self.df['signal'].apply(lambda x: scipy.signal.resample(x, 2000, axis=0))
		self.indexes = indexes
		print(df[target].sum(), df[target].sum() / len(df))
		
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		signal = self.df['signal'].iloc[idx]
		label = self.df[self.target].iloc[idx]
		signal =  np.array([smoothing(i) for i in signal])[self.indexes]
		signal = torch.Tensor(signal)
		return signal, label
	
	
if __name__ == '__main__':
	# норма [1:-1]
	# нижний [1:-1]
	# передний [8:10]
	# перегородочный [5:11]
	# передне-перегородочный [6:10]
	# передне-боковой [0, 1, 3, 9, 10, 11]
	
	# 1 (0.8) - (0.76, 0.78)
	# 2 (0.1) - (0.72, 0.8)
	# 3 (0.1) - (0.78, 0.82)
	
	indexes = {
		'норма': list(range(0, 12)),
		'нижний': list(range(1, 11)),
		'передний': [7, 8, 9, 10],
		'перегородочный': list(range(1, 11)),
		'передне-перегородочный': [6, 7, 8, 9],
		'передне-боковой': [0, 1, 3, 9, 10, 11],
	}
	
	models = {
		'норма': Model1DNorm,
		'нижний': Model1DDown,
		'передний': Model1DPered,
		'перегородочный': Model1DPereg,
		'передне-перегородочный': Model1DPeredPereg,
		'передне-боковой': Model1DSideDown,
	}
	
	
	target = 'норма'
	
	df = read_pickle('/Users/danil/AIIJC_FINAL/DATASETS/train_aug_full.pickle')
	df['kurtosis'] = df['signal'].apply(lambda x: scipy.stats.kurtosis(x, axis=1).mean())
	df = df.sort_values(by='kurtosis', ascending=False)
	df = df.iloc[:int(len(df) * 0.95)]
	
	
	df = df.merge(train_gts, on=['record_name'])
	
	df_norm = df[df['норма'] == 1]
	df_ab_norm = df[df['норма'] == 0]
	norm_r_names = list(set(df_norm['record_name']))
	ab_norm_r_names = list(set(df_ab_norm['record_name']))
	
	norm_r_names_train, norm_r_names_val = train_test_split(norm_r_names, test_size=0.25, random_state=25)
	ab_norm_r_names_train, ab_norm_r_names_val = train_test_split(ab_norm_r_names, test_size=0.25, random_state=25)
	
	train_norm = df_norm[df_norm['record_name'].isin(norm_r_names_train)]
	val_norm = df_norm[df_norm['record_name'].isin(norm_r_names_val)]
	
	train_ab_norm = df_ab_norm[df_ab_norm['record_name'].isin(ab_norm_r_names_train)]
	val_ab_norm = df_ab_norm[df_ab_norm['record_name'].isin(ab_norm_r_names_val)]
	
	train_dataset = Dataset1D(pd.concat([train_ab_norm, train_norm]), target=target, indexes=indexes[target])
	val_dataset = Dataset1D(pd.concat([val_ab_norm, val_norm]), target=target, indexes=indexes[target])
	
	filename = '/Users/danil/AIIJC_FINAL/PyCharm_code/MODELS/FullSignal/norm_conv.py'
	name = 'CONV_BiLSTM'
	# model = models[target]()
	
	model = CNNLstm(input_dim=len(indexes[target]))
	params = {
		'model': model,
		'train': train_dataset,
		'val': val_dataset,
		'epoches': 100,
		'lr': 0.0005,
		'batch_size': 32,
		'sh': 0.7,
		'milestones': [30],
		'verbose_tensorboard': True,
		'verbose_console': False,
		'num_workers': 0,
		'loss_weight': [1.0],
		'optimizer': AdamW,
		'name': f'FullSignal_model_{name}_' + target,
		'reverse_f1': True,
	}
	
	
	trainer = Trainer(**params)
	state_dict, best_f1, train_f1, best_iteration = trainer.train()
	print('BEST F1 VAL -', best_f1)
	print('BEST F1 TRAIN -', train_f1)
	print('BEST ITERATION -', best_iteration)

	filepath = '/Users/danil/AIIJC_FINAL/MODELS'
	if name not in os.listdir(filepath):
		os.mkdir(os.path.join(filepath, name))

	filepath = os.path.join(filepath, name)
	names_dict = {
		'норма': 'normal',
		'нижний': 'down',
		'передний': 'front',
		'перегородочный': 'septal',
		'передне-перегородочный': 'front_septal',
		'передне-боковой': 'front_down',
	}
	name = names_dict[target]
	if name not in os.listdir(filepath):
		os.mkdir(os.path.join(filepath, name))
	filepath = os.path.join(filepath, name)

	torch.save(state_dict, os.path.join(filepath, 'model_dict'))
	shutil.copyfile(filename, os.path.join(filepath, 'model.py'))