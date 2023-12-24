from torch import nn


class FrontSeptalV2(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
		self.relu_c1 = nn.ReLU()
		
		self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=2, stride=5)
		self.mp2 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm1d(64)
		self.relu_c2 = nn.ReLU()
		
		self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=2, stride=3)
		self.relu_c3 = nn.ReLU()
		self.bn3 = nn.BatchNorm1d(64)
		self.do2_3 = nn.Dropout2d(p=0.05)
		
		self.conv4 = nn.Conv1d(64, 64, kernel_size=7, padding=2, stride=2)
		self.relu_c4 = nn.ReLU()
		self.bn4 = nn.BatchNorm1d(64)
		
		self.conv5 = nn.Conv1d(64, 32, kernel_size=5, padding=2, stride=2)
		self.relu_c5 = nn.ReLU()
		self.mp5 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn5 = nn.BatchNorm1d(32)
		
		self.avg = nn.AvgPool1d(kernel_size=5, stride=2)
		
		self.linear1 = nn.Linear(192, 32)
		self.bn = nn.BatchNorm1d(32)
		self.relu_l1 = nn.ReLU()
		
		self.linear2 = nn.Linear(32, 32)
		self.relu_l2 = nn.ReLU()
		self.do1_2 = nn.Dropout(p=0.3)
		
		self.linear3 = nn.Linear(32, 32)
		self.relu_l3 = nn.ReLU()
		self.do1_3 = nn.Dropout(p=0.2)
		
		self.pred_final = nn.Linear(32, 32)
		self.relu_l4 = nn.ReLU()
		self.do1_4 = nn.Dropout(p=0.4)
		
		self.linear_final = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu_c1(x)
		
		x = self.conv2(x)
		x = self.relu_c2(x)
		x = self.mp2(x)
		
		x = self.conv3(x)
		x = self.relu_c3(x)
		x = self.bn3(x)
		x = self.do2_3(x)
		
		x = self.conv4(x)
		x = self.relu_c4(x)
		
		x = self.conv5(x)
		x = self.relu_c5(x)
		x = self.mp5(x)
		x = self.bn5(x)
		
		x = self.avg(x)
		x = x.view(x.shape[0], -1)
		
		x = self.linear1(x)
		x = self.relu_l1(x)
		x_save = x
		
		x = self.pred_final(x)
		x = self.relu_l4(x)
		x = self.do1_4(x)
		x += x_save
		
		x = self.linear_final(x)
		x = self.sigmoid(x)
		return x