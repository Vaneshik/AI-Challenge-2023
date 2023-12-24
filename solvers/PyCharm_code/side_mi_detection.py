import pandas as pd
from scipy.signal import resample
import torch
from torch import nn
from torch.utils.data import Dataset


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
		self.relu1 = nn.LeakyReLU()
		self.bn1 = nn.BatchNorm1d(16)
		self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
		self.relu2 = nn.LeakyReLU()
		self.do2 = nn.Dropout1d(p=0.2)
		self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
		self.relu3 = nn.LeakyReLU()
		self.do3 = nn.Dropout1d(p=0.1)
		self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
		self.relu4 = nn.SELU()
		self.bn4 = nn.BatchNorm1d(32)
		
		self.linear5 = nn.Linear(100, 64)
		self.relu5 = nn.ReLU()
		self.bn5 = nn.BatchNorm1d(64)
		self.linear6 = nn.Linear(64, 1)
		
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.do2(x)
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.do3(x)
		x = self.conv4(x)
		x = self.relu4(x)
		x = self.bn4(x)
		
		x = x.view(x.shape[0], -1)
		x = self.linear5(x)
		x = self.relu5(x)
		x = self.bn5(x)
		x = self.linear6(x)
		x = self.sigmoid(x)
		
		return x
	
	
class SideDataset(Dataset):
	def __init__(self, df: pd.DataFrame, alpha, is_train=True, new_length: int=200):
		df_true = df[df['боковой'] == 1]
		df_false = df[df['боковой'] == 0]
		df_false = df_false.sample(freq=1)
		df_false = df_false.iloc[:int(len(df_true) * alpha)]
		df_end = pd.concat([df_true, df_false])
		df_end['signal'] = df_end['signal'].apply(lambda x: resample(x, new_length))
		
		
		
		