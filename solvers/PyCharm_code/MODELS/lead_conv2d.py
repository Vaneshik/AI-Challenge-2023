from torch import nn, concat


class B(nn.Module):
	def __init__(self, in_, out_):
		super().__init__()
		
		self.intro_bn = nn.BatchNorm2d(32)
		
		self.intro = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=1)
		
		self.C11 = nn.Conv2d(32, 32, kernel_size=(3, 5), padding=1)
		self.A11 = nn.LeakyReLU()
		self.D11 = nn.Dropout2d(p=0.1)
		
		self.C12 = nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2)
		self.A12 = nn.ReLU()
		
		self.C13 = nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2)
		self.A13 = nn.ReLU()
		self.D13 = nn.Dropout2d(p=0.1)
		
		self.BN4 = nn.BatchNorm2d(32)
		self.M14 = nn.MaxPool2d(kernel_size=(7, 7), stride=1)
		
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
	def __init__(self, num_classes, leads=None, num_linear_layers: int=2):
		super().__init__()
		self.leads = leads
		self.d = {
			'I': 0, 'II': 1, 'III': 2, 'aVR': 3,
			'aVL': 4, 'aVF': 5, 'V1': 6, 'V2': 7,
			'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11,
		}
		
		in_, out_ = 8, 64
		self.conv_in = nn.Conv2d(1, in_, kernel_size=5)
		
		n = 12 if self.leads is None else len(self.leads)
		self.B_blocks = nn.ModuleList(B(in_, out_) for _ in range(n))
		self.conv_final_1 = nn.Conv2d(32, 8, kernel_size=(2, 5), padding=2)
		self.relu1 = nn.ReLU()
		self.conv_final_2 = nn.Conv2d(8, 8, kernel_size=(2, 5), padding=2)
		self.relu2 = nn.ReLU()
		self.mp = nn.MaxPool2d(kernel_size=2)
		
		self.avg_pool = nn.AvgPool2d(2)
		self.first_linear = nn.Linear(752, 32)
		self.first_relu = nn.LeakyReLU()
		self.linears = [nn.Linear(32, 32).to('mps') for _ in range(num_linear_layers)]
		self.linears = nn.ModuleList(self.linears)
		self.relus = [nn.ReLU() for _ in range(num_linear_layers)]
		self.f_linear = nn.Linear(32, num_classes)
		if num_classes > 1:
			self.softmax = nn.Softmax()
		else:
			self.softmax = nn.Sigmoid()
	
	def forward(self, x):
		if self.leads is None:
			res = [self.B_blocks[0](x[:, :, :7])]
			for i in range(1, len(self.B_blocks)):
				res.append(self.B_blocks[i](x[:, :, 7 * i:7 * (i + 1)]))
		else:
			res = []
			index = 0
			for lead in self.leads:
				idx = self.d[lead]
				res.append(self.B_blocks[index](x[:, :, 7 * idx:7 * (idx + 1)]))
				index += 1
		res = concat(res, axis=2)
		res = self.conv_final_1(res)
		res = self.relu1(res)
		res = self.conv_final_2(res)
		res = self.relu2(res)
		x = self.mp(res)
		x = self.avg_pool(x)
		x = x.view(x.shape[0], -1)
		print(x.shape)
		x = self.first_linear(x)
		x = self.first_relu(x)
		for i in range(len(self.linears)):
			x = self.linears[i](x)
			x = self.relus[i](x)
		x = self.f_linear(x)
		x = self.softmax(x)
		
		return x