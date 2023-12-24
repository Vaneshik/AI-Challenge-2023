import tqdm
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)


class Trainer:
	def __init__(self, generator, discriminator, train_dataset, val_dataset,
	             learning_rate: float, epochs: int, batch_size: int=32):
		self.device = 'mps'
		self.generator = generator.to(self.device)
		self.discriminator = discriminator.to(self.device)
		self.criterion = nn.BCELoss().to(self.device)
		self.epochs = epochs
		
		self.optimizerD = Adam(self.generator.parameters(), lr=learning_rate)
		self.optimizerG = Adam(self.discriminator.parameters(), lr=learning_rate)
		
		self.dataloaders = {
			'train': DataLoader(
				train_dataset,
				batch_size,
				shuffle=True,
			),
			'val': DataLoader(
				val_dataset,
				batch_size,
			)
		}
		
	def train_epoch(self, phase: str):
		if phase == 'train':
			self.discriminator.train()
			self.generator.train()
		else:
			self.discriminator.eval()
			self.generator.eval()
		discriminator_loss_arr = []
		generator_loss_arr = []

		for signal in self.dataloaders[phase]:
			signal = signal.to(self.device)
			signal = signal.unsqueeze(1)
			
			self.discriminator.zero_grad()
			batch_size = signal.shape[0]
			labels = torch.full((batch_size, 1), 1.0, device=self.device)
			
			out = self.discriminator(signal)
			real_loss = self.criterion(out, labels)
			real_loss.backward()

			noise = torch.randn(batch_size, 100, 1, device=self.device)
			fake_signal = self.generator(noise)
			labels.fill_(0.0)
			
			out = self.discriminator(fake_signal.detach())
			fake_loss = self.criterion(out, labels)
			fake_loss.backward()
			_ = fake_loss + real_loss
			self.optimizerD.step()
			
			self.generator.zero_grad()
			labels.fill_(1.0)
			out = self.discriminator(self.generator(noise))
			gen_loss = self.criterion(out, labels)
			gen_loss.backward()
			self.optimizerG.step()
			
			discriminator_loss = real_loss.mean().item() + fake_loss.mean().item()
			generator_loss = gen_loss.mean().item()
			discriminator_loss_arr.append(discriminator_loss)
			generator_loss_arr.append(generator_loss)
		
		return np.array(discriminator_loss_arr).mean(), np.array(generator_loss_arr).mean()
		
	
	
	def train(self):
		for epoch in tqdm(range(self.epochs)):
			for phase in ['train', 'val']:
				discriminator_loss, generator_loss = self.train_epoch(phase)
				print(f'EPOCH {epoch} {phase.upper()}: D - {discriminator_loss} ;; G - {generator_loss}')
	