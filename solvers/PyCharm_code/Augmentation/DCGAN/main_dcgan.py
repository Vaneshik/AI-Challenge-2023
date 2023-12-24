import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import functions
from constants import train_gts
from Trainer import Trainer
from Generator import Generator
from Discriminator import Discriminator


class GanDataset(Dataset):
	def __init__(self, signals: np.ndarray):
		self.signals = signals
		
	def __len__(self):
		return len(self.signals)
	
	def __getitem__(self, idx):
		return torch.Tensor(self.signals[idx])
	
	
if __name__ == '__main__':
	d = functions.read_pickle('/Users/danil/AIIJC_FINAL/DATASETS/train_frame')
	d = d.merge(train_gts, on=['record_name'])
	d['quantile_50'] = d['quantile_50'].apply(lambda x: scipy.signal.resample(x, 354))
	signals = np.array(d[(d['норма'] == 1) & (d['group'] == 0)]['quantile_50'].tolist())
	
	train_signals, val_signals = train_test_split(signals, test_size=0.1, random_state=25)
	train_dataset, val_dataset = GanDataset(train_signals), GanDataset(val_signals)
	
	generator, discriminator = Generator(100), Discriminator()
	
	trainer = Trainer(
		generator, discriminator, train_dataset, val_dataset,
		0.0001, 10,
	)
	trainer.train()
	device = 'mps'
	for i in range(10):
		noice = torch.randn(100, 1, device=device)
		result = generator(noice)
		np.save(f'/Users/danil/AIIJC_FINAL/DATASETS/GAN/dcgan{i}.npy', result.view(-1).detach().cpu().numpy())
	