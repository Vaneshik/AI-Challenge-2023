from torch import nn


class Conv1d(nn.Module):
	def __init__(self, input_dim, output_dim, kernel: int = 6, padding: int = 0, stride: int = 0):
		super().__init__()
		self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel, padding=padding, stride=stride)
	
	def forward(self, x):
		x = self.conv(x)
		return x


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		# self.L1 = Conv1d(1, 64, 4, 1, 2)
		# self.L2 = Conv1d(64, 128, 4, 1, 2)
		# self.L3 = Conv1d(128, 256, 4, 1, 2)
		# self.L4 = Conv1d(256, 128, 4, 1, 2)
		# self.L5 = Conv1d(128, 64, 4, 1, 2)
		
		self.L1 = nn.Conv1d(1, 64, kernel_size=4, padding=1, stride=2)
		self.L2 = nn.Conv1d(64, 128, kernel_size=4, padding=1, stride=2)
		self.L3 = nn.Conv1d(128, 256, kernel_size=4, padding=1, stride=2)
		self.L4 = nn.Conv1d(256, 128, kernel_size=4, padding=1, stride=2)
		self.L5 = nn.Conv1d(128, 64, kernel_size=4, padding=1, stride=2)
		
		self.L6 = nn.Linear(704, 1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		x = self.L1(x)
		x = self.L2(x)
		x = self.L3(x)
		x = self.L4(x)
		x = self.L5(x)
		x = x.view(x.shape[0], -1)
		x = self.L6(x)
		x = self.sigmoid(x)
		return x