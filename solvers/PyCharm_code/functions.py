import tqdm
import numpy as np
import pandas as pd
import pickle
import neurokit2 as nk


def read_pickle(path: str) -> pd.DataFrame:
	with open(path, mode='rb') as file:
		return pickle.load(file)


def dump_pickle(path: str, obj: pd.DataFrame):
	with open(path, mode='wb') as file:
		pickle.dump(obj, file)


def read_signal(path: str):
	with open(path, mode='rb') as file:
		return np.load(path, allow_pickle=True)
	
	
def clean_signal(signal: np.array):
	new_signal = []
	for s in signal:
		try:
			s_proc = nk.ecg_clean(s, sampling_rate=500)
			new_signal.append(s_proc[500:-500])
		except Exception:
			new_signal.append(np.zeros(4000))
	return np.array(new_signal)

	