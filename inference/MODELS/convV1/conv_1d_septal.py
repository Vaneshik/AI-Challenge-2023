from torch import nn


class SeptalV1(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=3)
		self.relu_c1 = nn.ReLU()
		
		self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2)
		self.relu_c2 = nn.ReLU()
		
		self.conv3 = nn.Conv1d(64, 64, kernel_size=9, padding=3, stride=3)
		self.relu_c3 = nn.ReLU()
		self.do2_3 = nn.Dropout2d(p=0.11)
		
		self.conv4 = nn.Conv1d(64, 64, kernel_size=7, padding=3, stride=2)
		self.relu_c4 = nn.ReLU()
		self.bn4 = nn.BatchNorm1d(64)
		
		self.conv5 = nn.Conv1d(64, 32, kernel_size=7, padding=3, stride=3)
		self.relu_c5 = nn.ReLU()
		self.do2_5 = nn.Dropout2d(p=0.11)
		
		self.conv6 = nn.Conv1d(32, 16, kernel_size=7, padding=3, stride=2)
		self.relu_c6 = nn.ReLU()
		self.do2_6 = nn.Dropout2d(p=0.11)
		self.bn6 = nn.BatchNorm1d(16)
		
		self.conv7 = nn.Conv1d(16, 16, kernel_size=10, padding=3, stride=2)
		self.relu_c7 = nn.ReLU()
		self.do2_7 = nn.Dropout2d(p=0.1)
		
		self.avgpool = nn.AvgPool1d(5, padding=2, stride=1)
		
		self.linear1 = nn.Linear(896, 32)
		self.relu_l1 = nn.ReLU()
		
		self.linear2 = nn.Linear(32, 32)
		self.relu_l2 = nn.ReLU()
		self.do1_2 = nn.Dropout(p=0.1)
		
		self.linear3 = nn.Linear(32, 32)
		self.relu_l3 = nn.ReLU()
		self.do1_3 = nn.Dropout(p=0.1)
		
		self.linear4 = nn.Linear(32, 32)
		self.relu_l4 = nn.ReLU()
		self.do1_4 = nn.Dropout(p=0.1)
		
		self.linear_final = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu_c1(x)
		
		x_save = x
		x = self.conv2(x)
		x = self.relu_c2(x)
		
		x = self.conv3(x)
		x = self.relu_c3(x)
		x = self.do2_3(x)
		
		x = self.conv4(x)
		x = self.relu_c4(x)
		# x = self.bn4(x)
		
		x = self.conv5(x)
		x = self.relu_c5(x)
		x = self.do2_5(x)
		
		x = self.conv6(x)
		x = self.relu_c6(x)
		x = self.do2_6(x)
		x = self.avgpool(x)
		
		x = x.view(x.shape[0], -1)
		
		x = self.linear1(x)
		x = self.relu_l1(x)
		x_save = x
		
		# x = self.linear2(x)
		# x = self.relu_l2(x)
		# x = self.do1_2(x)
		
		x = self.linear3(x)
		x = self.relu_l3(x)
		x = self.do1_3(x)
		# x += x_save
		
		x = self.linear_final(x)
		x = self.sigmoid(x)
		return x