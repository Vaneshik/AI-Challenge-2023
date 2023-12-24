import torch
import numpy as np
from torch import nn


class B(nn.Module):
	def __init__(self, in_, out_):
		super().__init__()
		
		self.C11 = nn.Conv2d(in_, in_, kernel_size=(3, 24), padding=(1, 5))
		self.A11 = nn.LeakyReLU()
		self.D11 = nn.Dropout2d(p=0.1)
		
		self.C13 = nn.Conv2d(in_, out_, kernel_size=(5, 40), padding=(2, 5))
		self.A13 = nn.ReLU()
		
		self.BN4 = nn.BatchNorm2d(out_)
		self.M14 = nn.MaxPool2d(kernel_size=(3, 24), stride=1)
		
		self.D14 = nn.Dropout2d(p=0.1)
	
	def forward(self, x):
		x = self.C11(x)
		x = self.A11(x)
		x = self.D11(x)
		
		x = self.C13(x)
		x = self.A13(x)
		
		x = self.BN4(x)
		x = self.M14(x)
		x = self.D14(x)
		
		return x


class LeadLayersModel(nn.Module):
	def __init__(self, length, device='mps'):
		super().__init__()
		self.length = length
		self.conv1 = [B(1, 32).to(device) for _ in range(12)]
		self.conv2 = [B(32, 16).to(device) for _ in range(4)]
		self.conv3 = B(16, 8).to(device)
		self.avg = nn.AvgPool2d(2)
		
		self.linear1 = nn.Linear(400, 64)
		self.relu1 = nn.LeakyReLU()
		self.linear2 = nn.Linear(64, 16)
		self.relu2 = nn.ReLU()
		self.do2 = nn.Dropout(p=0.1)
		self.linear3 = nn.Linear(16, 1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		results = [[] for _ in range(len(self.conv2))]
		k = len(self.conv1) // len(self.conv2)
		for i in range(len(self.conv1)):
			results[i // k].append(self.conv1[i](x[:, :, i * self.length:(i + 1) * self.length]))
		results = [torch.concat(i, axis=2) for i in results]
		results2 = []
		for i in range(len(self.conv2)):
			results2.append(self.conv2[i](results[i]))
		results2 = torch.concat(results2, axis=2)
		results2 = self.conv3(results2)
		results2 = self.avg(results2)
		
		results2 = results2.view(results2.shape[0], -1)
		print(results2.shape)
		
		results2 = self.linear1(results2)
		results2 = self.relu1(results2)
		results2 = self.linear2(results2)
		results2 = self.relu2(results2)
		results2 = self.do2(results2)
		results2 = self.linear3(results2)
		results2 = self.sigmoid(results2)
		return results2