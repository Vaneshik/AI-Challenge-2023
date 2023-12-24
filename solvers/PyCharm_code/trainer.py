import os

import tqdm
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Any

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR,
                                      MultiStepLR)

from metrics import Metrics

import warnings
import shutil
warnings.filterwarnings("ignore")


class Trainer:
	def __init__(self, model, train, val, epoches: int=50,
	             lr: float=0.0001, batch_size: int=20, sh: float=5e-4, milestones: list[int]=[5, 10], number: int=0,
	             num_workers: int=0, loss_weight: list[float]=[3.0],
	             optimizer=Adam, name=None,
	             verbose_tensorboard: bool=True, verbose_console: bool=True, reverse_f1: bool=True):
		self.num_workers = num_workers
		self.number = number
		self.epoches = epoches
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.reverse_f1 = reverse_f1
		
		self.device = torch.device("mps")
		
		log_dir = path.join('runs', name) if name is not None else None
		if log_dir is not None:
			try:
				shutil.rmtree(log_dir)
			except FileNotFoundError:
				pass
		self.torch_writer = SummaryWriter(log_dir=log_dir)
		
		self.model = model.to(self.device)
		self.optim = optimizer(self.model.parameters(), lr=lr, weight_decay=5e-4)
		# self.scheduler = CosineAnnealingLR(self.optim, T_max=epoches, eta_min=sh)
		self.scheduler = MultiStepLR(self.optim, gamma=sh, milestones=milestones)
		self.criterion = nn.BCELoss(weight=torch.tensor(loss_weight).to(self.device))
		
		self.verbose_tensorboard = verbose_tensorboard
		self.verbose_console = verbose_console
		
		self.metrics_functions: dict[str, Callable[[Any, Any], float]] = {
			'f1-score': (lambda x, y: f1_score(x, y)) if not reverse_f1 else (lambda x, y: f1_score(x * -1 + 1, y * -1 + 1)),
			# 'accuracy': accuracy_score,
			# 'roc_auc': roc_auc_score,
		}
		
		self.idx = self.train_idx = self.val_idx = 0
		
		self.dataloaders = {
			'train': DataLoader(
				train,
				batch_size=batch_size,
				num_workers=num_workers,
				shuffle=False),
			'val': DataLoader(
				val,
				batch_size=batch_size,
				num_workers=num_workers,
				shuffle=False)
		}
		
	
	def _train_epoch(self, phase: str) -> tuple[dict[str, float], list[float], list[int]]:
		metrics = Metrics(self.metrics_functions)
		
		self.model.train() if phase == 'train' else self.model.eval()
		all_x = []
		all_y = []
		for signals, labels in self.dataloaders[phase]:
			signals = signals.to(self.device)
			labels = labels.type(torch.FloatTensor).to(self.device)
			# signals = torch.unsqueeze(signals, 1)
			labels = torch.unsqueeze(labels, 1)
			
			out = self.model(signals)
			loss = self.criterion(out, labels)
			
			if phase == 'train':
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

			metrics.update(out, labels, loss)
			
			x = list(np.round(out.cpu().detach().numpy()))
			y = list(labels.cpu().detach().numpy())
			all_x += x
			all_y += y

		return metrics.get_means(), all_x, all_y
	
	def train(self) -> tuple[nn.Module, float, float, int]:
		metrics = {
			'train': Metrics(self.metrics_functions),
			'val': Metrics(self.metrics_functions)
		}
		best_f1 = 0.0
		best_model = None
		best_iteration = 0
		train_f1 = 0.0
		best_train_f1 = 0.0
		
		for epoch in tqdm(range(self.epoches)):
			if self.verbose_console:
				print(f'EPOCH [ {epoch} ]')
			for phase in ['train', 'val']:
				result_metrics, x, y = self._train_epoch(phase)
				# metrics[phase].soft_update(loss, score)
				for k, v in result_metrics.items():
					if k == 'loss':
						self.torch_writer.add_scalar(f'{k}/{phase}', v, epoch)
				if self.reverse_f1:
					y = [1 - i for i in y]
					x = [1 - i for i in x]
				print(sum(x) / len(x))
				f1 = f1_score(y, x)
				if phase == 'train':
					train_f1 = f1
				self.torch_writer.add_scalar(f'f1-score/{phase}', f1, epoch)
				if phase == 'val' and f1 > best_f1:
					best_f1 = f1
					best_model = self.model.state_dict()
					best_iteration = epoch
					best_train_f1 = train_f1
				
				# if self.verbose_console:
				# 	print(f'{phase.upper()} : LOSS - {loss}\tF1-score - {score}')
			self.scheduler.step()
			# torch.save(self.model, '/Users/danil/Downloads/model' + '_' + str(self.number) + '_' + str(epoch))
		self.idx += 1
		
		return best_model, best_f1, best_train_f1, best_iteration
