from torch import nn


class NormV2(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=1)
		self.relu_c1 = nn.ReLU()
		
		self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=1, stride=2)
		self.bn2 = nn.BatchNorm1d(64)
		self.relu_c2 = nn.ReLU()
		
		self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=1, stride=1)
		self.relu_c3 = nn.ELU()
		self.mp3 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm1d(64)
		self.do2_3 = nn.Dropout2d(p=0.2)
		
		self.conv4 = nn.Conv1d(64, 64, kernel_size=7, padding=2, stride=3)
		self.relu_c4 = nn.ReLU()
		self.bn4 = nn.BatchNorm1d(64)
		
		self.conv5 = nn.Conv1d(64, 64, kernel_size=7, padding=2, stride=1)
		self.relu_c5 = nn.ReLU()
		self.mp5 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn5 = nn.BatchNorm1d(64)
		
		self.conv6 = nn.Conv1d(64, 64, kernel_size=7, padding=1, stride=2)
		self.relu_c6 = nn.ELU()
		self.bn6 = nn.BatchNorm1d(64)
		self.do2_6 = nn.Dropout2d(p=0.2)
		
		self.conv7 = nn.Conv1d(64, 64, kernel_size=5, padding=1, stride=1)
		self.relu_c7 = nn.ReLU()
		self.mp7 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn7 = nn.BatchNorm1d(64)
		
		self.conv8 = nn.Conv1d(64, 32, kernel_size=5, padding=1, stride=3)
		self.relu_c8 = nn.ELU()
		self.bn8 = nn.BatchNorm1d(32)
		self.do2_8 = nn.Dropout2d(p=0.15)
		
		self.conv9 = nn.Conv1d(32, 32, kernel_size=5, padding=1, stride=1)
		self.relu_c9 = nn.ReLU()
		self.mp9 = nn.MaxPool1d(kernel_size=5, stride=2)
		self.bn9 = nn.BatchNorm1d(32)
		
		self.avg = nn.AvgPool1d(kernel_size=5, stride=2)
		
		self.linear1 = nn.Linear(96, 32)
		self.relu_l1 = nn.ELU()
		self.bn_lin = nn.BatchNorm1d(32)
		self.do1_1 = nn.Dropout(p=0.2)
		
		self.linear2 = nn.Linear(32, 32)
		self.relu_l2 = nn.ELU()
		self.do1_2 = nn.Dropout(p=0.2)
		
		self.linear3 = nn.Linear(32, 32)
		self.relu_l3 = nn.ReLU()
		self.do1_3 = nn.Dropout(p=0.2)
		
		self.pred_final = nn.Linear(32, 32)
		self.relu_l4 = nn.ELU()
		self.do1_4 = nn.Dropout(p=0.1)
		
		self.linear_final = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu_c1(x)
		
		x = self.conv2(x)
		x = self.relu_c2(x)
		x = self.bn2(x)
		
		x = self.conv3(x)
		x = self.relu_c3(x)
		x = self.mp3(x)
		x = self.bn3(x)
		x = self.do2_3(x)
		
		x = self.conv4(x)
		x = self.relu_c4(x)
		x = self.bn4(x)
		
		x = self.conv5(x)
		x = self.relu_c5(x)
		x = self.bn5(x)
		
		x = self.conv6(x)
		x = self.relu_c6(x)
		x = self.bn6(x)
		x = self.do2_6(x)
		
		x = self.conv7(x)
		x = self.relu_c7(x)
		x = self.mp7(x)
		x = self.bn7(x)
		
		x = self.conv8(x)
		x = self.relu_c8(x)
		x = self.bn8(x)
		x = self.do2_8(x)
		
		x = self.conv9(x)
		x = self.relu_c9(x)
		x = self.mp9(x)
		x = self.bn9(x)
		
		x = self.avg(x)
		x = x.view(x.shape[0], -1)
		
		x = self.linear1(x)
		x = self.relu_l1(x)
		x_save = x
		x = self.bn_lin(x)
		x = self.do1_1(x)
		
		x = self.linear2(x)
		x = self.relu_l2(x)
		x = self.do1_2(x)
		#
		# x = self.linear3(x)
		# x = self.relu_l3(x)
		# x = self.do1_3(x)
		#
		x = self.pred_final(x)
		x = self.relu_l4(x)
		x = self.do1_4(x)
		x += x_save
		
		x = self.linear_final(x)
		x = self.sigmoid(x)
		return x