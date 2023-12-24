import torch
from torch import nn, concat


class A(nn.Module):
	def __init__(self, old_value, value, kernel, padding, is_max_pool: bool=True):
		super().__init__()
		self.conv1 = nn.Conv1d(old_value, value, kernel_size=kernel, padding=padding)
		self.conv2 = nn.Conv1d(value, value, kernel_size=kernel, padding=padding)
		self.conv3 = nn.Conv1d(value, value, kernel_size=kernel, padding=padding)
		self.max_pool = nn.MaxPool1d(kernel_size=kernel, padding=padding)
		self.is_max_pool = is_max_pool
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		if self.is_max_pool:
			x = self.max_pool(x)
		return x
	
	
class B(nn.Module):
	def __init__(self):
		super().__init__()
		self.L1 = A(1, 4, 3, 1)
		self.L2 = A(4, 8, 5, 2)
		self.L3 = A(8, 16, 5, 2, False)
		self.AvgPool = nn.AvgPool1d(2, padding=1)
	
	def forward(self, x):
		x = self.L1(x)
		x = self.L2(x)
		x = self.L3(x)
		return x
	
	
class LeadConv1d(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = [B().to('mps') for _ in range(12)]
		self.d = nn.Dropout1d(0.1)
		self.lstm1 = nn.Linear(2688, 32)
		self.lstm2 = nn.Linear(32, 32)
		self.relu1 = nn.ReLU()
		self.linear1 = nn.Linear(32, 16)
		self.relu2 = nn.LeakyReLU()
		self.linear2 = nn.Linear(16, 1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		res = []
		for i in range(12):
			res.append(self.layers[i](x[:, :, 7 * i + 2]))
		res = torch.concat(res, axis=2)
		res = self.d(res)
		res = res.view(res.shape[0], -1)
		# print(res.shape)
		res = self.lstm1(res)
		res = self.lstm2(res)
		res = self.relu1(res)
		res = self.linear1(res)
		res = self.relu2(res)
		res = self.linear2(res)
		res = self.sigmoid(res)
		return res
		
	
		