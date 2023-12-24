import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from typing import Callable, Any

import warnings
warnings.filterwarnings("ignore")


class Metrics:
	def __init__(self, metrics_functions: dict[str, Callable[[Any, Any], float]]):
		"""
		metrics_functions: dict[name of metric, function of metric (signature - [true_y, pred_y])]
		"""
		
		self.metrics_functions = metrics_functions
		self.metrics_dict: dict[str, list[float]] = dict([(k, []) for k in self.metrics_functions.keys()])
		self.metrics_dict['loss'] = []
	
	def soft_update(self, loss: float, score: dict[str, float]):
		self.metrics_dict['loss'].append(loss)
		assert set(score.keys()) == set(self.metrics_dict.keys()) - {'loss'}
		for k, v in score.items():
			self.metrics_dict[k].append(v)
	
	def update(self, x, y, loss):
		loss = loss.item()
		score = {}
		if x.shape[1] == 1:
			x = np.round(x.cpu().detach().numpy())
			y = y.cpu().detach().numpy()
			for k, func in self.metrics_functions.items():
				score[k] = func(y, x)
		else:
			x = np.round(x.cpu().detach().numpy())
			y = y.cpu().detach().numpy()
			for k, func in self.metrics_functions.items():
				for i in range(x.shape[1]):
					score[k] = func(y[:, i], x[:, i])
				score[k] /= x.shape[1]
		
		self.soft_update(loss, score)
	
	def get_means(self) -> dict[str, float]:
		means_dict = {}
		for k, v in self.metrics_dict.items():
			means_dict[k] = sum(v) / len(v)
		return means_dict
	
	def __str__(self):
		ans = ''
		for k, v in self.get_means().items():
			ans += f'{k.upper()} - {v}\n'
		return ans
	
	def graphic(self) -> None:
		_, axis = plt.subplots(1, len(self.metrics_dict), figsize=(20, 10))
		idx = 0
		for k, v in self.metrics_dict:
			axis[idx].plot(v)
			axis[idx].set_title(k)
			idx += 1
		plt.plot()