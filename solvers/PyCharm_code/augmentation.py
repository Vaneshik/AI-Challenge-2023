import tqdm
import numpy as np
from scipy.signal import resample
from constants import TRAIN_PATH, train_gts
from tqdm.auto import tqdm
from functions import read_signal
from os import path, listdir, remove
import random


def scale(signal: np.ndarray) -> np.ndarray:
	k: float = np.random.choice([0.3, 0.5, 0.8, 1.2, 1.7])
	return signal * k


def add_some_noise(signal: np.ndarray) -> np.ndarray:
	noise = np.random.normal(0, np.random.uniform(0.02, 0.04), (12, 5000))
	return signal + noise


def time_wrapping(signal: np.ndarray) -> np.ndarray:
	length = signal.shape[1]
	wrapping_index = np.random.randint(int(length * 0.4), int(length * 0.6))
	k: float = np.random.uniform(1.2, 1.5)
	if np.random.random() < 0.5:
		k = 1 / k
	
	signal_left, signal_right = signal[:, :wrapping_index], signal[:, wrapping_index:]
	signal_left = np.array([resample(i, int(wrapping_index * k)) for i in signal_left])
	signal_right = np.array([resample(i, length - int(wrapping_index * k)) for i in signal_right])
	signal = np.concatenate([signal_left, signal_right], axis=1)
	return signal


def shift(signal: np.ndarray) -> np.ndarray:
	length = signal.shape[1]
	shift_scale = np.random.randint(int(length * 0.1), int(length * 0.9))
	return np.roll(signal, shift_scale, axis=1)


def augmentation(signal: np.ndarray, label: str) -> tuple[list[np.ndarray], list[str]]:
	augmentation_functions = [
		('scale', scale),
		('tw', time_wrapping),
		('noice', add_some_noise),
		('shift', shift),
	]
	variables = [(0, 1, 2, 3),
	             (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
	             (0, 1), (0, 2), (1, 2), (1,), (2,)]
	
	n: int
	if label != 'норма':
		n = np.random.randint(2, 4)
	else:
		n = np.random.randint(4, 6)
	n = 0
	
	result_signals = []
	result_names = []
	for _ in range(n):
		function_indexes = random.choice(variables)
		s = signal.copy()
		final_name = ''
		for idx in function_indexes:
			name, f = augmentation_functions[idx]
			s = f(s)
			final_name += name + '_'
		final_name = final_name[:-1]
		result_signals.append(s)
		result_names.append(final_name)
	
	return result_signals, result_names


dirpath = '/Users/danil/AIIJC_FINAL/DATASETS/augmentation/train_augmentation'
input_path = TRAIN_PATH
dataset = train_gts

r_names = dataset['record_name']
label_names = ['норма', 'нижний', 'передний', 'перегородочный', 'передне-перегородочный', 'передне-боковой']
labels = dataset[label_names].to_numpy()

for file in listdir(dirpath):
	remove(path.join(dirpath, file))
	
for name_idx in tqdm(range(len(r_names))):
	r_name = r_names[name_idx]
	signal = read_signal(path.join(TRAIN_PATH, r_name + '.npy'))
	target_label: str = ''
	for i in range(len(label_names)):
		if labels[name_idx, i] == 1:
			target_label = label_names[i]
			break
	
	np.save(path.join(dirpath, r_name + '.npy'), signal)
	if target_label == 'норма':
		continue
	aug_signals, aug_names = augmentation(signal, target_label)
	for i in range(len(aug_names)):
		np.save(path.join(dirpath, r_name + f'_aug{i}_' + aug_names[i] + '.npy'), aug_signals[i])
		
print(len(listdir(dirpath)) * 224 / 1024 / 1024, 'Gb')
		