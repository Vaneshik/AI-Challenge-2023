from torch import nn


class Convolution(nn.Module):
	def __init__(self, input_dim, output_dim, kernel=5, stride=1):
		super().__init__()
		self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel, stride=stride)
		self.bn = nn.BatchNorm1d(output_dim)
		self.do = nn.Dropout(p=0.15)
		
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.do(x)
		return x


class CNNLstm(nn.Module):
	def __init__(self, input_dim: int):
		super().__init__()
		self.conv1 = Convolution(input_dim, 32, stride=1)
		self.conv2 = Convolution(32, 64, stride=3)
		self.conv3 = Convolution(64, 64, stride=3)
		self.conv4 = Convolution(64, 64, stride=3)
		self.conv5 = Convolution(64, 64, stride=3)
		self.conv6 = Convolution(64, 64, stride=3)
		
		
		self.avg = nn.AvgPool1d(2)
		self.linear1 = nn.Linear(448, 128)
		self.relu1 = nn.ReLU()
		self.linear2 = nn.Linear(128, 32)
		self.relu2 = nn.ReLU()
		self.linear3 = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		
		x = self.avg(x)
		x = x.view(x.shape[0], -1)
		x = self.linear1(x)
		x = self.relu1(x)
		x = self.linear2(x)
		x = self.relu2(x)
		x = self.sigmoid(self.linear3(x))
		return x
		