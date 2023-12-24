import torch
from torch import nn


class SqueezeExcitation(nn.Module):
	def __init__(self, input_channels, squeeze_channels):
		super(SqueezeExcitation, self).__init__()
		self.avgpool = torch.nn.AvgPool2d(1)
		self.fc1 = torch.nn.Conv1d(input_channels, squeeze_channels, 1)
		self.fc2 = torch.nn.Conv1d(squeeze_channels, input_channels, 1)
		self.activation = nn.ReLU()
		self.scale_activation = nn.Sigmoid()
	
	def scale(self, x):
		scale = self.avgpool(x)
		scale = self.fc1(scale)
		scale = self.activation(scale)
		scale = self.fc2(scale)
		return self.scale_activation(scale)
	
	def forward(self, x):
		scale = self.scale(x)
		return scale * x


class ConvolutionBlock(nn.Module):
	def __init__(self, _in, _out):
		super().__init__()
		self.conv = nn.Conv1d(_in, _out, 3, 1)
		self.bn = nn.BatchNorm1d(_out)
		self.elu = nn.ELU()
		self.maxpool = nn.MaxPool1d(3)
	
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.elu(x)
		x = self.maxpool(x)
		
		return x


class SEBlock(nn.Module):
	def __init__(self, _in, squeeze_channels):
		super().__init__()
		space = 2 * _in
		self.conv1 = nn.Conv1d(_in, space, 1)
		self.bn1 = nn.BatchNorm1d(space)
		self.elu1 = nn.ELU()
		self.conv2 = nn.Conv1d(space, _in, 1)
		self.bn2 = nn.BatchNorm1d(_in)
		self.se = SqueezeExcitation(_in, squeeze_channels)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.elu1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.se(x)
		
		return x


class Block(nn.Module):
	def __init__(self, _in, space, squeeze_channels):
		super().__init__()
		self.conv_block = ConvolutionBlock(_in, space)
		self.se_block = SEBlock(space, squeeze_channels)
		self.elu = nn.ELU()
	
	def forward(self, x):
		x0 = self.conv_block(x)
		x1 = self.se_block(x0)
		x1 += x0
		x1 = self.elu(x1)
		return x1


class Se_Resnet(nn.Module):
	def __init__(self, num_classes, space_1=128, space_2=128, squeeze_channels=256):
		super(Se_Resnet, self).__init__()
		self.block1 = Block(num_classes, space_1, squeeze_channels)
		self.block2 = Block(space_1, space_2, squeeze_channels)
		
		self.avg = nn.AdaptiveAvgPool1d(1)
		self.flatten = nn.Flatten()
		
		self.relu = nn.ReLU()
		self.linear1 = nn.Linear(space_2, space_2)
		self.linear2 = nn.Linear(32, 32)
		
		self.fc = nn.Linear(space_2, 1)
		self.sig = nn.Sigmoid()
	
	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		
		x = self.avg(x)
		x = self.flatten(x)
		
		x = self.linear1(x)
		x = self.relu(x)
		x = self.fc(x)
		x = self.sig(x)
		return x