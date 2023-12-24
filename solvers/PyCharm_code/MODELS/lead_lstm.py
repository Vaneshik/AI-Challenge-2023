from torch import nn, concat


class B(nn.Module):
	def __init__(self, in_, out_):
		super().__init__()
		
		self.intro_bn = nn.BatchNorm2d(32)
		
		self.intro = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		
		self.C11 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		self.A11 = nn.LeakyReLU()
		self.D11 = nn.Dropout2d(p=0.1)
		
		self.C12 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
		self.A12 = nn.ReLU()
		
		self.C13 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
		self.A13 = nn.ReLU()
		self.D13 = nn.Dropout2d(p=0.1)
		
		self.BN4 = nn.BatchNorm2d(32)
		self.M14 = nn.MaxPool2d(kernel_size=7, stride=2)
		
		self.D14 = nn.Dropout2d(p=0.15)
	
	def forward(self, x):
		x = self.intro(x)
		x = self.intro_bn(x)
		
		x = self.C11(x)
		x = self.A11(x)
		x = self.D11(x)
		
		x = self.C12(x)
		x = self.A12(x)
		
		x = self.C13(x)
		x = self.A13(x)
		x = self.D13(x)
		
		x = self.BN4(x)
		x = self.M14(x)
		x = self.D14(x)
		
		return x


class LeadLSTM(nn.Module):
	def __init__(self):
		super().__init__()
		
		in_, out_ = 8, 64
		self.conv_in = nn.Conv2d(1, in_, kernel_size=5)
		
		self.B_blocks = nn.ModuleList(B(in_, out_) for _ in range(12))
		self.conv_final_1 = nn.Conv2d(32, 8, kernel_size=(5, 5), padding=1)
		self.relu = nn.ReLU()
		self.mp = nn.MaxPool2d(kernel_size=2)
		
		self.avg_pool = nn.AvgPool2d(2)
		self.fc0_lstm = nn.LSTM(368, 256)
		self.bn0 = nn.BatchNorm1d(256)
		self.acc0 = nn.LeakyReLU()
		self.fc1_lstm = nn.LSTM(256, 32)
		self.bn1 = nn.BatchNorm1d(32)
		self.acc1 = nn.LeakyReLU()
		self.fc2 = nn.Linear(32, 1)
		self.sigm = nn.Sigmoid()
	
	def forward(self, x):
		res = [self.B_blocks[0](x[:, :, :7])]
		for i in range(1, len(self.B_blocks)):
			res.append(self.B_blocks[i](x[:, :, 7 * i:7 * (i + 1)]))
		res = concat(res, axis=2)
		res = self.conv_final_1(res)
		res = self.relu(res)
		x = self.mp(res)
		x = self.avg_pool(x)
		x = x.view(x.shape[0], -1)
		print(x.shape)
		x, _ = self.fc0_lstm(x)
		x = self.bn0(x)
		x = self.acc0(x)
		x, _ = self.fc1_lstm(x)
		x = self.bn1(x)
		x = self.acc1(x)
		x = self.fc2(x)
		x = self.sigm(x)
		
		return x