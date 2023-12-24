import random
import time

import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, SparseAdam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR,
                                      MultiStepLR)

from sklearn.model_selection import train_test_split

from scipy.signal import resample
from scipy.ndimage import median_filter

import scipy

from trainer import Trainer
from constants import train_meta, train_gts
from MODELS.lead_conv2d import LeadModel
from MODELS.straight_conv2d import StraightModel
from MODELS.lead_straight_model import MergeModel
from MODELS.lead_lstm import LeadLSTM
from MODELS.lead_conv1d import LeadConv1d
from MODELS.LeadMultiLabels import LeadMultiLabels
from MODELS.lead_conv2d_layers import LeadLayersModel

# import constants

import warnings
warnings.filterwarnings("ignore")


def moving_avg(x, n):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[n:] - cumsum[:-n]) / float(n)


def process_dataset(df: pd.DataFrame, new_length: int=200) -> pd.DataFrame:
	def smoothing(signal: np.ndarray) -> np.ndarray:
		for i in range(2, 5):
			signal = moving_avg(signal, i)
		return signal
	
	def smoothing_median(signal: np.ndarray) -> np.ndarray:
		for _ in range(6):
			signal = median_filter(signal, 3)
		return signal
	
	def crop_signal(signal: np.ndarray) -> np.ndarray:
		return signal[25:-70]
	
	def signal_to_new_length(signal: np.ndarray, new_length: int) -> np.ndarray:
		return resample(signal,  new_length)
	
	def norm(signal: np.ndarray) -> np.ndarray:
		return (signal - signal.mean()) / (signal.std())
	
	df['signal'] = df['signal'].apply(smoothing)
	df['signal'] = df['signal'].apply(crop_signal)
	df['signal'] = df['signal'].apply(lambda x: signal_to_new_length(x, new_length))
	# df['signal'] = df['signal'].apply(norm)
	if 'kurtosis' not in df.columns:
		df['kurtosis'] = df['signal'].apply(scipy.stats.kurtosis)
	df = df.sort_values(by=['kurtosis'], ascending=False)
	return df


def get_group_signal_stats(signals: np.ndarray) -> list[np.ndarray]:
	# return list(np.linspace(signals.min(axis=0), signals.max(axis=0), 7))
	ret = [
		np.quantile(signals, 0.3, axis=0),
		signals.std(axis=0),
		np.quantile(signals, 0.5, axis=0),
		np.quantile(signals, 0.7, axis=0),
		np.tile(np.quantile(signals, 0.5, axis=0)[30:-120], 4),
		np.quantile(signals, 0.7, axis=0) - np.quantile(signals, 0.3, axis=0),
		np.quantile(signals, 0.5, axis=0) - np.roll(np.quantile(signals, 0.5, axis=0), -1),
	]
	return ret


def get_2d_signals(df: pd.DataFrame, target_name: str='норма', is_train: bool=True) -> tuple[np.ndarray, np.ndarray]:
	signals = []
	labels = []
	
	for name in list(list(sorted(list(set(df['record_name']))))):
		if 'aug' in name:
			continue
		current: pd.DataFrame = df[df['record_name'] == name]
		if len(set(current['group'])) < 12 and is_train:
			continue
		if is_train:
			label = current.iloc[0][target_name]
			try:
				labels.append(label.tolist())
			except TypeError:
				labels.append(label)
		else:
			labels.append(name)
		s: list[np.ndarray] = []
		for group_idx in range(12):
			current_group = current[current['group'] == group_idx]
			if len(current_group) > 0:
				current_group = current_group.iloc[:int(len(current_group) * 0.9) + 1]
				current_signals = np.array(current_group['signal'].tolist())
				s += get_group_signal_stats(current_signals)
			else:
				s += signals[-1][group_idx * 7:(group_idx + 1) * 7]
		signals.append(s)
	return np.array(labels), np.array(signals)


class MyDataset(Dataset):
	def __init__(self, df, is_train, new_length: int=200, group_skip=False, target_name: str='норма') -> None:
		self.is_train = is_train
		df = process_dataset(df, new_length)
		self.labels, signals = get_2d_signals(df, target_name=target_name, is_train=is_train)
		print(signals.shape)
		self.signals = torch.Tensor(signals)
	
	def __len__(self) -> int:
		return len(self.signals)
	
	def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
		if self.is_train:
			return self.signals[idx], self.labels[idx]
		return self.signals[idx], -1
	
	
# class DatasetFrame(Dataset):
# 	def __init__(self, df: pd.DataFrame, is_train=False):
# 		self.labels: np.ndarray
# 		self.is_train = is_train
# 		columns = ['record_name', 'group']
# 		if is_train:
# 			self.labels = np.array(df['myocard'].tolist())[::12]
# 			columns += ['myocard']
# 		self.signals = df.drop(columns=columns).to_numpy()
# 		self.signals = self.signals.reshape(self.signals.shape[0] // 12, -1)
# 		self.signals = np.array([[list(i) for i in j] for j in self.signals])
#
# 	def __len__(self) -> int:
# 		return len(self.signals)
#
# 	def __getitem__(self, idx):
# 		if self.is_train:
# 			return torch.Tensor(self.signals[idx]), self.labels[idx]
# 		return torch.Tensor(self.signals[idx]), -1
#
#
# if __name__ == '__main__':
# 	with open('/Users/danil/AIIJC_FINAL/DATASETS/train_frame', mode='rb') as file:
# 		df_cleaned = pickle.load(file)
# 	names = list(set(df_cleaned['record_name']))
# 	train_names, val_names = train_test_split(names, test_size=0.2, random_state=25)
# 	train_df = df_cleaned[df_cleaned['record_name'].isin(train_names)]
# 	val_df = df_cleaned[df_cleaned['record_name'].isin(val_names)]
#
# 	train_dataset, val_dataset = DatasetFrame(train_df, True), DatasetFrame(val_df, True)
# 	model = LeadModel()
# 	params = {
# 		'model': model,
# 		'train': train_dataset,
# 		'val': val_dataset,
# 		'epoches': 20,
# 		'lr': 0.00001,
# 		'batch_size': 32,
# 		'sh': 5e-4,
# 		'verbose_tensorboard': True,
# 		'verbose_console': False,
# 		'num_workers': 0,
# 		'loss_weight': 4.0,
# 		'optimizer': AdamW,
# 		'name': 'LeadModel_AdamW_0d00001_200_frame'
# 	}
#
# 	t = Trainer(**params)
# 	t.train()
	
	
	

if __name__ == '__main__':
	with open('/Users/danil/AIIJS_PROJECT/clean_datasets/clean_train.pickle', mode='rb') as file:
		df_cleaned = pickle.load(file)
	
	st = time.time()
	names = list(set(df_cleaned['record_name']))
	trash_names = ['15857_hr.npy', '12629_hr.npy', '11813_hr.npy', '12916_hr.npy',
       '21601_hr.npy', '03777_hr.npy', '00930_hr.npy', '16376_hr.npy',
       '08456_hr.npy', '07265_hr.npy', '16767_hr.npy', '10301_hr.npy',
       '15760_hr.npy', '16585_hr.npy', '06664_hr.npy', '08964_hr.npy',
       '21170_hr.npy', '12432_hr.npy', '00144_hr.npy', '02983_hr.npy',
       '18099_hr.npy', '21193_hr.npy', '17098_hr.npy', '06154_hr.npy',
       '02587_hr.npy', '07442_hr.npy', '04791_hr.npy', '04377_hr.npy',
       '19118_hr.npy', '11982_hr.npy', '14734_hr.npy', '09557_hr.npy',
       '19096_hr.npy', '00385_hr.npy', '09781_hr.npy', '00527_hr.npy',
       '08530_hr.npy', '13507_hr.npy', '01391_hr.npy', '20513_hr.npy',
       '01399_hr.npy', '03412_hr.npy', '08343_hr.npy', '08524_hr.npy',
       '03087_hr.npy', '14173_hr.npy', '02247_hr.npy', '14718_hr.npy',
       '21753_hr.npy', '17830_hr.npy', '21487_hr.npy']
	# names = list(filter(lambda x: x not in trash_names, names))
	df_cleaned = df_cleaned.merge(train_meta[['record_name', 'strat_fold']], on=['record_name'])
	df_cleaned = df_cleaned.merge(train_gts, on=['record_name'])
	# df_cleaned = df_cleaned[df_cleaned['myocard'] == 1]
	train_names, val_names = train_test_split(names, test_size=0.2, random_state=25)
	# train_df = df_cleaned[df_cleaned['record_name'].isin(train_names)]
	# val_df = df_cleaned[df_cleaned['record_name'].isin(val_names)]
	
	df = df_cleaned[df_cleaned['норма'] == 0]
	train_df = df[df['strat_fold'].isin([1, 2, 3, 5, 6, 7, 9, 10])]
	val_df = df[df['strat_fold'].isin([4, 8])]
	
	target = 'нижний'
	# train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	#                               MyDataset(val_df, True, new_length=200, target_name=target))
	
	
	# variance = [
	# 	(0.0001, 250, AdamW, 'LeadModel', 'LeadModel_AdamW_0d0001_250'),
	# 	(0.000095, 250, AdamW, 'LeadModel', 'LeadModel_AdamW_0d000095_250'),
	# 	(0.00009, 250, AdamW, 'LeadModel', 'LeadModel_AdamW_0d00009_250'),
	# 	(0.000015, 250, AdamW, 'LeadModel', 'LeadModel_AdamW_0d00015_250'),
	# 	(0.00002, 250, AdamW, 'LeadModel', 'LeadModel_AdamW_0d0002_250'),
	# ]
	
	# for target in ['перегородочный', 'передний', 'боковой',
    #                            'передне-боковой', 'передне-перегородочный',
    #                            'нижний']:
	# 	df = df_cleaned[df_cleaned['норма'] == 0]
	# 	train_df = df[df['strat_fold'].isin([1, 2, 3, 5, 6, 7, 9, 10])]
	# 	val_df = df[df['strat_fold'].isin([4, 8])]
	# 	train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	# 	                              MyDataset(val_df, True, new_length=200, target_name=target))
	# 	model = LeadModel(1)
	# 	params = {
	# 		'model': model,
	# 		'train': train_dataset,
	# 		'val': val_dataset,
	# 		'epoches': 15,
	# 		'lr': 0.0004,
	# 		'batch_size': 32,
	# 		'sh': 0.77,
	# 		'verbose_tensorboard': True,
	# 		'verbose_console': False,
	# 		'num_workers': 0,
	# 		'loss_weight': [6.0],
	# 		'optimizer': AdamW,
	# 		'name': 'LeadModel_AdamW_0d0001_200_stratfold_mult_predict_' + target,
	# 	}
	#
	# 	t = Trainer(**params)
	# 	t.train()
	#
	# 	df = df_cleaned[(df_cleaned['норма'] == 1) | (df_cleaned[target] == 1)]
	# 	train_df = df[df['strat_fold'].isin([1, 2, 3, 5, 6, 7, 9, 10])]
	# 	val_df = df[df['strat_fold'].isin([4, 8])]
	# 	train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	# 	                              MyDataset(val_df, True, new_length=200, target_name=target))
	# 	# params['name'] = 'LeadModel_AdamW_0d0001_200_stratfold_mult_' + target + '_norm'
	# 	# params['train'] = train_dataset
	# 	# params['val'] = val_dataset
	# 	params = {
	# 		'model': model,
	# 		'train': train_dataset,
	# 		'val': val_dataset,
	# 		'epoches': 30,
	# 		'lr': 0.001,
	# 		'batch_size': 32,
	# 		'sh': 0.86,
	# 		'verbose_tensorboard': True,
	# 		'verbose_console': False,
	# 		'num_workers': 0,
	# 		'loss_weight': [7.0],
	# 		'optimizer': AdamW,
	# 		'name': 'LeadModel_AdamW_0d0003_200_stratfold_mult_predict_' + target + '_norm'
	# 	}
	#
	# 	t = Trainer(**params)
	# 	t.train()
	#
	# 	torch.save(model, '/Users/danil/AIIJC_FINAL/MODELS/Model_AdamW_predict_' + target)
	# print(time.time() - st)
	target = 'передний'
	# df = df_cleaned[df_cleaned['норма'] == 0]
	# [1, 2, 3, 5, 6, 7, 9, 10], [5, 9]
	df = df_cleaned
	train_df = df[df['strat_fold'].isin([1, 2, 3, 4, 6, 7, 8, 10])]
	val_df = df[df['strat_fold'].isin([5, 9])]
	
	train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	                              MyDataset(val_df, True, new_length=200, target_name=target))
	model = LeadModel(1, ['V3', 'V4'], 1)
	# model = LeadModel(1, None, 0)
	# model = LeadLayersModel(7)
	params = {
		'model': model,
		'train': train_dataset,
		'val': val_dataset,
		'epoches': 50,
		'lr': 0.002,
		'batch_size': 32,
		'sh': 0.83,
		'milestones': [4, 12, 20],
		'verbose_tensorboard': True,
		'verbose_console': False,
		'num_workers': 0,
		'loss_weight': [8.0],
		'optimizer':AdamW,
		'name': 'LeadModel_AdamW_0d0008_0linears_test_' + target,
		'reverse_f1': False,
	}

	t = Trainer(**params)
	t.train()
	
	torch.save(model, '/Users/danil/AIIJC_FINAL/MODELS/Model_AdamW_predict_poop' + target)

	# df = df_cleaned[(df_cleaned['норма'] == 1) | (df_cleaned[target] == 1)]
	# train_df = df[df['strat_fold'].isin([1, 2, 3, 5, 6, 7, 9, 10])]
	# val_df = df[df['strat_fold'].isin([4, 8])]
	# train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	#                               MyDataset(val_df, True, new_length=200, target_name=target))
	# # params['name'] = 'LeadModel_AdamW_0d0001_200_stratfold_mult_' + target + '_norm'
	# # params['train'] = train_dataset
	# # params['val'] = val_dataset
	# params = {
	# 	'model': model,
	# 	'train': train_dataset,
	# 	'val': val_dataset,
	# 	'epoches': 30,
	# 	'lr': 0.003,
	# 	'batch_size': 32,
	# 	'sh': 0.99,
	# 	'verbose_tensorboard': True,
	# 	'verbose_console': False,
	# 	'num_workers': 0,
	# 	'loss_weight': [10.0],
	# 	'optimizer': AdamW,
	# 	'name': 'LeadModel_AdamW_0d0003_200_stratfold_mult_' + target + '_norm'
	# }
	# # params['loss_weight'] = 6.0
	#
	# t = Trainer(**params)
	# t.train()
	
	# with open('/Users/danil/AIIJC_FINAL/DATASETS/mixup_abnormal_dataset.pickle', mode='rb') as file:
	# 	df_cleaned = pickle.load(file)
	#
	# target = [
	# 	'перегородочный',
	# 	'передний',
	# 	'боковой',
	# 	'передне-боковой',
	# 	'передне-перегородочный',
	# 	'нижний'
	# ]
	
	# names = list(set(df_cleaned['record_name']))
	# train_names, val_names = train_test_split(names, test_size=0.2)
	# df = df_cleaned
	# train_df = df[df['record_name'].isin(train_names)]
	# val_df = df[df['record_name'].isin(val_names)]
	# print(train_df.shape)
	# train_dataset, val_dataset = (MyDataset(train_df, True, new_length=200, target_name=target),
	#                               MyDataset(val_df, True, new_length=200, target_name=target))
	#
	# model = LeadMultiLabels(6)
	# params = {
	# 	'model': model,
	# 	'train': train_dataset,
	# 	'val': val_dataset,
	# 	'epoches': 30,
	# 	'lr': 0.0001,
	# 	'batch_size': 64,
	# 	'sh': 0.99,
	# 	'verbose_tensorboard': True,
	# 	'verbose_console': False,
	# 	'num_workers': 0,
	# 	'loss_weight': [4.0, 2.0, 2.0, 3.0, 2.0, 1.0],
	# 	'optimizer': AdamW,
	# 	'name': 'LeadModel_AdamW_0d0003_200_stratfold_multiclass'
	# }
	# # params['loss_weight'] = 6.0
	#
	# t = Trainer(**params)
	# t.train()