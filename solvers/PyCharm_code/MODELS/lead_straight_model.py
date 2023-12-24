from torch import nn, concat
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
	def __init__(self):
		super().__init__()
		
		in_, out_ = 32, 64
		self.conv_in = nn.Conv2d(1, in_, kernel_size=5)
		
		self.A_blocks = nn.ModuleList(A(in_, out_) for _ in range(5))
		self.conv_final = nn.Conv2d(32, 32, kernel_size=7, padding=2)
		
		self.avg_pool = nn.AvgPool2d(2)
		self.fc1 = nn.Linear(576, 128)
		self.acc1 = nn.ReLU()
	
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
		
		return x


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


class LeadModel(nn.Module):
	def __init__(self):
		super().__init__()
		
		in_, out_ = 8, 64
		self.conv_in = nn.Conv2d(1, in_, kernel_size=5)
		
		self.B_blocks = nn.ModuleList(B(in_, out_) for _ in range(12))
		self.conv_final_1 = nn.Conv2d(32, 8, kernel_size=(5, 5), padding=1)
		self.relu = nn.ReLU()
		self.mp = nn.MaxPool2d(kernel_size=2)
		
		self.avg_pool = nn.AvgPool2d(2)
		self.fc0 = nn.Linear(368, 128)
		self.acc0 = nn.LeakyReLU()
	
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
		x = self.fc0(x)
		x = self.acc0(x)
		
		return x
	
	
class MergeModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.straight_layer = StraightModel()
		self.lead_layer = LeadModel()
		
		self.linear_1 = nn.Linear(256, 128)
		self.relu_1 = nn.ReLU()
		self.linear_2 = nn.Linear(128, 1)
		self.relu_2 = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		straight_res = self.straight_layer(x)
		lead_res = self.lead_layer(x)
		res = concat([straight_res, lead_res], axis=1)
		res = self.linear_1(res)
		res = self.relu_1(res)
		res = self.linear_2(res)
		res = self.relu_2(res)
		res = self.sigmoid(res)
		return res
		