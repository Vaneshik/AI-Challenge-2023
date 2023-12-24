from torch import nn


class Conv1d(nn.Module):
	def __init__(self, input_dim, output_dim, kernel: int=6, padding: int=0, stride: int=1):
		super().__init__()
		self.conv = nn.ConvTranspose1d(input_dim, output_dim, kernel_size=kernel, padding=padding, stride=stride)
		
	def forward(self, x):
		x = self.conv(x)
		return x


class Generator(nn.Module):
	def __init__(self, noise_dim: int):
		super().__init__()
		# self.L1 = Conv1d(noise_dim, 2048).to('mps')
		# self.L2 = Conv1d(2048, 1028).to('mps')
		# self.L3 = Conv1d(1028, 512, stride=2, padding=1, kernel=4).to('mps')
		# self.L4 = Conv1d(512, 256, stride=2, padding=1, kernel=4).to('mps')
		# self.L5 = Conv1d(256, 128, stride=2, padding=1, kernel=4).to('mps')
		# self.L6 = Conv1d(128, 64, stride=2, padding=1, kernel=4).to('mps')
		# self.L7 = Conv1d(64, 1, stride=2, padding=1, kernel=6).to('mps')
		
		self.L1 = nn.ConvTranspose1d(noise_dim, 2048, kernel_size=6, padding=0, stride=1)
		self.L2 = nn.ConvTranspose1d(2048, 1028, kernel_size=6, padding=0, stride=1)
		self.L3 = nn.ConvTranspose1d(1028, 512, kernel_size=4, padding=1, stride=2)
		self.L4 = nn.ConvTranspose1d(512, 256, kernel_size=4, padding=1, stride=2)
		self.relu = nn.ReLU()
		self.L5 = nn.ConvTranspose1d(256, 128, kernel_size=4, padding=1, stride=2)
		self.L6 = nn.ConvTranspose1d(128, 64, kernel_size=4, padding=1, stride=2)
		self.L7 = nn.ConvTranspose1d(64, 1, kernel_size=6, padding=1, stride=2)
		# 260
		
	def forward(self, x):
		x = self.L1(x)
		x = self.L2(x)
		x = self.L3(x)
		x = self.L4(x)
		x = self.relu(x)
		x = self.L5(x)
		x = self.L6(x)
		x = self.L7(x)
		return x
		
		
		
		