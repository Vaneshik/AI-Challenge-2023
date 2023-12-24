from torch import nn

import warnings
warnings.filterwarnings("ignore")


class A(nn.Module):
	def __init__(self, in_, out_):
		super().__init__()
		
		self.intro_bn = nn.BatchNorm2d(in_)
		
		self.C11 = nn.Conv2d(in_, out_, kernel_size=5, padding=2)
		self.A11 = nn.ReLU()
		self.C12 = nn.Conv2d(out_, in_, kernel_size=5, padding=2)
		self.A12 = nn.ReLU()
		self.M11 = nn.MaxPool2d(kernel_size=5, stride=2)
		
		self.D11 = nn.Dropout2d(p=0.08)
	
	def forward(self, x):
		x = self.intro_bn(x)
		C = x
		x = self.C11(x)
		x = self.A11(x)
		x = self.C12(x)
		x = x + C
		x = self.A12(x)
		x = self.M11(x)
		x = self.D11(x)
		
		return x


class StraightModel(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		
		in_, out_ = 32, 64
		self.conv_in = nn.Conv2d(1, in_, kernel_size=5)
		
		self.A_blocks = nn.ModuleList(A(in_, out_) for _ in range(5))
		self.conv_final = nn.Conv2d(32, 32, kernel_size=7, padding=2)
		
		self.avg_pool = nn.AvgPool2d(2)
		self.fc1 = nn.Linear(576, 32)
		self.acc1 = nn.ReLU()
		self.fc2 = nn.Linear(32, num_classes)
		self.softmax = nn.Softmax()
	
	def forward(self, x):
		x = self.conv_in(x)
		
		for i in range(3):
			x = self.A_blocks[i](x)
		
		x = self.conv_final(x)
		x = self.avg_pool(x)
		x = x.view(x.shape[0], -1)
		print(x.shape)
		x = self.fc1(x)
		x = self.acc1(x)
		x = self.fc2(x)
		x = self.softmax(x)
		
		return x