import torch
from torch import nn
import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import pickle
import warnings
import time
import scipy
from catboost import CatBoostClassifier

from MODELS.convV1.conv_1d_down import DownV1
from MODELS.convV1.conv_1d_norm import NormV1
from MODELS.convV1.conv_1d_front import FrontV1
from MODELS.convV1.conv_1d_septal import SeptalV1
from MODELS.convV1.conv_1d_front_down import FrontDownV1
from MODELS.convV1.conv_1d_front_septal import FrontSeptalV1

from MODELS.convV2.conv_1d_down_v2 import DownV2
from MODELS.convV2.conv_1d_norm_v2 import NormV2
from MODELS.convV2.conv_1d_front_v2 import FrontV2
from MODELS.convV2.conv_1d_septal_v2 import SeptalV2
from MODELS.convV2.conv_1d_front_down_v2 import FrontDownV2
from MODELS.convV2.conv_1d_front_septal_v2 import FrontSeptalV2

from MODELS.se_resnet import Se_Resnet
from MODELS.skip_connected_conv import CNN

warnings.simplefilter('ignore')


# https://github.com/Vaneshik/AIChallengeBot GitHub of bot:
# https://t.me/minions_AI_Challenge_Bot telegram link on bot


class Config:
	translate_eng_ru = {
		'down': 'нижний',
		'front': 'передний',
		'front_down': 'передне-боковой',
		'front_septal': 'передне-перегородочный',
		'normnal': 'норма',
		'septal': 'перегородочный',
		'side': 'боковой'
	}
	translate_ru_eng = dict([(j, i) for i, j in translate_eng_ru.items()])
	v1_shapes = {
		'down': list(range(1, 11)),
		'front': [8, 9],
		'front_down': [0, 1, 3, 9, 10, 11],
		'front_septal': [6, 7, 8, 9],
		'normal': list(range(1, 11)),
		'septal': [5, 6, 7, 8, 9, 10],
	}
	other_shapes = {
		'down': list(range(1, 11)),
		'front': [7, 8, 9, 10],
		'front_down': [0, 1, 3, 9, 10, 11],
		'front_septal': [6, 7, 8, 9],
		'normal': list(range(0, 12)),
		'septal': list(range(1, 11)),
	}
	v1_models = {
		'down': DownV1,
		'normal': NormV1,
		'front': FrontV1,
		'septal': SeptalV1,
		'front_down': FrontDownV1,
		'front_septal': FrontSeptalV1,
	}
	v2_models = {
		'down': DownV2,
		'normal': NormV2,
		'front': FrontV2,
		'septal': SeptalV2,
		'front_down': FrontDownV2,
		'front_septal': FrontSeptalV2,
	}
	sc_models_hid_size = {
		'down': 64,
		'front': 64,
		'septal': 80,
		'front_down': 100,
		'front_septal': 70,
	}
	target = [
		'перегородочный',
	    'передний',
	    'боковой',
	    'передне-боковой',
	    'передне-перегородочный',
	    'нижний',
	    'норма'
	]
	
	weights_path: str = 'weights'


def read_pickle(path: str) -> pd.DataFrame:
	with open(path, mode='rb') as file:
		return pickle.load(file)


def dump_pickle(path: str, obj: pd.DataFrame):
	with open(path, mode='wb') as file:
		pickle.dump(obj, file)


def read_signal(path: str):
	with open(path, mode='rb') as file:
		return np.load(path, allow_pickle=True)


def moving_avg(x, n):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[n:] - cumsum[:-n]) / float(n)


def smoothing(signal: np.ndarray, right: int) -> np.ndarray:
	for i in range(2, right):
		signal = moving_avg(signal, i)
	return signal


def clean_signal(signal: np.array, _type: int):
	new_signal = []
	for s in signal:
		try:
			s_proc = nk.ecg_clean(s, sampling_rate=500)
			s_proc = s_proc[500:-500]
			if _type == 0:
				s_proc = scipy.signal.medfilt(s_proc, 3)
				s_proc = smoothing(s_proc, 4)
			else:
				s_proc = smoothing(s_proc, 5)
			new_signal.append(s_proc)
		except Exception:
			new_signal.append(s[500:-500])
	return np.array(new_signal)


activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	
	return hook


def print_warning(msg: str) -> None:
	warnings.simplefilter('default', category=UserWarning)
	warnings.warn(msg)
	warnings.simplefilter('ignore', category=UserWarning)


def get_dataset(filepath: str) -> pd.DataFrame:
	"""
	Формат входного датасета должен соответствовать условию (одна папка где лежат сигналы и meta-данные
	"""
	signals_0, signals_1, r_names = [], [], []
	meta_data = None
	meta_columns = ['age', 'sex', 'height', 'weight', 'record_name']
	for name in os.listdir(filepath):
		fullname = os.path.join(filepath, name)
		if name.endswith('csv') and 'meta' in name:
			meta_data = pd.read_csv(fullname)
		if name.endswith('npy'):
			signal = read_signal(fullname)
			signals_0.append(clean_signal(signal, 0))
			signals_1.append(clean_signal(signal, 1))
			r_names.append(name[:-4])
	
	dataset = pd.DataFrame({
		'record_name': r_names,
		'signal_0': signals_0,
		'signal_1': signals_1,
	})
	if meta_data is None:
		print_warning("В указанной папке нет метаданных пациентов, рекомендуем это исправит,"
		              " иначе результат может быть менее точный")
		meta_data = pd.DataFrame([[None] * len(meta_columns) for _ in range(len(dataset))], columns=meta_columns)
	if 'record_name' not in meta_data.columns:
		meta_data['record_name'] = r_names
	for column in meta_columns:
		if column not in meta_data.columns:
			print_warning(f'В метаданных пациентов нет столбца "{column}"')
			meta_data[column] = [None] * len(dataset)
	meta_data = meta_data[meta_columns]
	dataset = dataset.merge(meta_data, on=['record_name'])
	return dataset


def get_signals(dataset: pd.DataFrame, device: str, _type: int) -> torch.Tensor:
	signals = torch.Tensor(np.array(dataset[f'signal_{_type}'].tolist()))
	signals = signals.to(device)
	return signals


def init_models(network_type: str, device: str) -> dict[str, nn.Module]:
	"""
	network types:
		V1: ConvV1:
		V2: ConvV2:
		se: Se_Resnet:
		sc: CNN
	"""
	assert network_type in ['V1', 'V2', 'se', 'sc']
	models: dict[str, nn.Module] = {}
	if network_type in ['V1', 'V2']:
		prefix = 'conv' + network_type + '_'
		models_dict = Config.v1_models if network_type == 'V1' else Config.v2_models
		for name, model in models_dict.items():
			models[name] = model()
			state_dict_name = prefix + name
			state_dict_path = os.path.join(Config.weights_path, state_dict_name)
			models[name].load_state_dict(torch.load(state_dict_path, map_location=device))
			models[name].eval()
	elif network_type == 'se':
		prefix = 'se_resnet_'
		for name, shape in Config.other_shapes.items():
			state_dict_name = prefix + name
			state_dict_path = os.path.join(Config.weights_path, state_dict_name)
			models[name] = Se_Resnet(num_classes=len(shape))
			models[name].load_state_dict(torch.load(state_dict_path, map_location=device))
			models[name].eval()
	else:
		prefix = 'sc_'
		for name, hid_size in Config.sc_models_hid_size.items():
			state_dict_name = prefix + name
			state_dict_path = os.path.join(Config.weights_path, state_dict_name)
			models[name] = CNN(input_size=len(Config.other_shapes[name]), hid_size=hid_size)
			models[name].load_state_dict(torch.load(state_dict_path, map_location=device))
			models[name].eval()
	
	# for name in models.keys():
	# 	models[name] = models[name].to(device)
	
	return models


def predict_conv1v(dataset: pd.DataFrame, device: str) -> pd.DataFrame:
	signals = get_signals(dataset, device, 0)
	
	# initialize models
	models: dict[str, nn.Module] = init_models('V1', device)
	
	ans_dataframe = pd.DataFrame({'record_name': dataset['record_name']})
	for name, model in models.items():
		res = []
		for chunk_signal in torch.chunk(signals, len(signals)):
			layer_name: str = ''
			if name == 'normal':
				layer_name = 'linear4'
				model.linear4.register_forward_hook(get_activation(layer_name))
			else:
				layer_name = 'linear3'
				model.linear3.register_forward_hook(get_activation(layer_name))
			
			_ = model(chunk_signal[:, Config.v1_shapes[name]])
			active = activation[layer_name].detach().cpu().numpy()
			res += list(active)
		ans_dataframe['model_conv_1d_' + name] = res
	
	return ans_dataframe


def predict_conv2v(dataset: pd.DataFrame, device: str) -> pd.DataFrame:
	signals = get_signals(dataset, device, 0)
	
	# initialize models
	models: dict[str, nn.Module] = init_models('V2', device)
	
	ans_dataframe = pd.DataFrame({'record_name': dataset['record_name']})
	for name, model in models.items():
		res = []
		for chunk_signal in torch.chunk(signals, len(signals)):
			layer_name = 'pred_final'
			model.pred_final.register_forward_hook(get_activation(layer_name))
			_ = model(chunk_signal[:, Config.other_shapes[name]])
			active = activation[layer_name].detach().cpu().numpy()
			res += list(active)
		ans_dataframe['model_conv_2d_' + name] = res
	
	return ans_dataframe


def predict_se_resnet(dataset: pd.DataFrame, device: str) -> pd.DataFrame:
	signals = get_signals(dataset, device, 1)
	
	# initialize models
	models: dict[str, nn.Module] = init_models('se', device)
	
	ans_dataframe = pd.DataFrame({'record_name': dataset['record_name']})
	for name, model in models.items():
		res = []
		for chunk_signal in torch.chunk(signals, len(signals)):
			layer_name: str = 'linear1'
			model.linear1.register_forward_hook(get_activation(layer_name))
			_ = model(chunk_signal[:, Config.other_shapes[name]])
			active = activation[layer_name].detach().cpu().numpy()
			res += list(active)
		ans_dataframe['model_se_resnet_' + name] = res
	
	return ans_dataframe


def predict_sc(dataset: pd.DataFrame, device: str) -> pd.DataFrame:
	signals = get_signals(dataset, device, 1)
	
	# initialize models
	models: dict[str, nn.Module] = init_models('sc', device)
	
	ans_dataframe = pd.DataFrame({'record_name': dataset['record_name']})
	for name, model in models.items():
		res = []
		for chunk_signal in torch.chunk(signals, len(signals)):
			layer_name: str = 'avgpool'
			model.avgpool.register_forward_hook(get_activation(layer_name))
			_ = model(chunk_signal[:, Config.other_shapes[name]])[0][0]
			active = activation[layer_name].detach().cpu().squeeze(2).numpy()
			res += list(active)
		ans_dataframe['model_sc_resnet_' + name] = res
	
	return ans_dataframe


def parse_data_from_penultimate_layers(dataset: pd.DataFrame, k: int) -> pd.DataFrame:
	result_dataset = pd.DataFrame()
	for column in dataset.columns:
		if column == 'record_name':
			continue
		if not column.startswith('model'):
			result_dataset[column] = dataset[column]
			continue

		data = np.array(dataset[column].tolist())
		for i in range(data.shape[1]):
			if i % (data.shape[1] // k) != 0:
				continue
			result_dataset[f'{column}_{i}'] = list(data[:, i])
	return result_dataset


def my_features(ecg_signal):
	SAMPLING_RATE = 500
	
	r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE, correct_artifacts=True)
	ecg_rate = nk.ecg_rate(r_peaks, sampling_rate=SAMPLING_RATE)
	ecg_vars = [np.mean(ecg_rate), np.min(ecg_rate), np.max(ecg_rate), np.max(ecg_rate) - np.min(ecg_rate)]
	hrv_columns = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
	               'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN',
	               'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN',
	               'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN',
	               'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_TP']
	
	try:
		hrv_time = nk.hrv_time(r_peaks[0], sampling_rate=SAMPLING_RATE)
		hrv_freq = nk.hrv_frequency(r_peaks[0], sampling_rate=SAMPLING_RATE)
		hrv = hrv_time.join(hrv_freq)
		hrv = hrv[hrv_columns].iloc[0].tolist()
	except Exception:
		print('minus signal')
		hrv = [None] * len(hrv_columns)
	
	entropy = nk.entropy_sample(ecg_signal, 1, 4)[0]
	features = ecg_vars + hrv + [entropy]
	
	return features


def get_features(dataset: pd.DataFrame) -> pd.DataFrame:
	hrv_columns = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
	               'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN',
	               'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN',
	               'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN',
	               'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_TP']
	columns = ['RateMean', 'RateMin', 'RateMax', 'RateRaz', ]
	columns += hrv_columns
	columns += ['Entopy']
	results = []
	for s in dataset['signal_0']:
		result = my_features(s[8])
		results.append(result)
	results = pd.DataFrame(results, columns=columns)
	results['record_name'] = dataset['record_name']
	return dataset.merge(results, on=['record_name'])


def catboost_predict(filepaths: dict[str, str], data):
	models = {}
	for k, v in filepaths.items():
		model = CatBoostClassifier()
		model.load_model(v)
		models[k] = model
	proba = pd.DataFrame()
	for k, v in models.items():
		proba[k] = v.predict_proba(data)[:, 1]
	
	return proba


def get_ans(proba: pd.DataFrame, r_names: list[str]) -> pd.DataFrame:
	trashes = {
		'перегородочный': 0.4,
		'передний': 0.75,
		'передне-боковой': 0.75,
		'передне-перегородочный': 0.5,
		'нижний': 0.65,
	}
	proba.loc[proba['боковой'] > 0.49, 'боковой'] = 1
	answer = pd.DataFrame()
	for t in Config.target:
		answer[t] = [0] * len(proba)
	answer.loc[proba['боковой'] > 0.49, 'боковой'] = 1
	answer.loc[(proba['норма'] == proba.max(axis=1)) | (proba['норма'] > 0.5), 'норма'] = 1
	for t in Config.target:
		if t == 'боковой':
			continue
		answer.loc[(proba.max(axis=1) < 0.5) & (proba.max(axis=1) == proba[t]), t] = 1
	for t in Config.target:
		if t == 'норма' or t == 'боковой':
			continue
		answer.loc[(proba[t] >= trashes[t]) & (proba['норма'] != proba.max(axis=1)) & (proba['норма'] <= 0.5), t] = 1
	
	answer['record_name'] = r_names
	
	return answer
	

def predict(dataset: pd.DataFrame, display_time=False):
	start_time = time.time()
	# you could use mps
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	result_dataset = dataset.copy()
	for func in [predict_conv1v, predict_conv2v, predict_sc, predict_se_resnet]:
		result = func(dataset, device)
		result_dataset = result_dataset.merge(result, on=['record_name'])
	
	result_dataset = get_features(result_dataset)
	result_dataset = result_dataset.drop(columns=['signal_0', 'signal_1'])
	
	r_names = result_dataset['record_name']
	result_dataset_10 = parse_data_from_penultimate_layers(result_dataset, 10)
	result_dataset_13 = parse_data_from_penultimate_layers(result_dataset, 13)
	result_dataset_16 = parse_data_from_penultimate_layers(result_dataset, 16)
	catboosts_10 = dict([(i, os.path.join(Config.weights_path, f'catboost_10_{Config.translate_ru_eng[i]}')) for i in Config.target])
	catboosts_13 = dict([(i, os.path.join(Config.weights_path, f'catboost_13_{Config.translate_ru_eng[i]}')) for i in Config.target])
	catboosts_16 = dict([(i, os.path.join(Config.weights_path, f'catboost_16_{Config.translate_ru_eng[i]}')) for i in Config.target])
	
	proba_10 = catboost_predict(catboosts_10, result_dataset_10)
	proba_13 = catboost_predict(catboosts_13, result_dataset_13)
	proba_16 = catboost_predict(catboosts_16, result_dataset_16)
	proba = 0.15 * proba_10 + 0.7 * proba_13 + 0.15 * proba_16
	answer = get_ans(proba, r_names)
	
	wall_time = time.time() - start_time
	if display_time:
		print('WALL TIME:', wall_time)
	
	return answer


def predict_inference(filepath: str, display_time=False) -> pd.DataFrame:
	dataset = get_dataset(filepath)
	result_dataset = predict(dataset, display_time=display_time)
	return result_dataset


YOUR_PATH = ''
answer = predict_inference(YOUR_PATH, display_time=True)
dump_pickle(os.path.join(YOUR_PATH, 'answer'), answer)


# answer = predict_inference('/Users/danil/AIIJC_FINAL/DATA/train', display_time=True)
# dump_pickle('/Users/danil/AIIJC_FINAL/DATASETS/last_predict_train.pickle', answer)


