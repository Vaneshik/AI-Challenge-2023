import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)


class ConvNormPool(nn.Module):
	"""Conv Skip-connection module"""
	
	def __init__(
			self,
			input_size,
			hidden_size,
			kernel_size,
			norm_type='bachnorm'
	):
		super().__init__()
		
		self.kernel_size = kernel_size
		self.conv_1 = nn.Conv1d(
			in_channels=input_size,
			out_channels=hidden_size,
			kernel_size=kernel_size
		)
		self.conv_2 = nn.Conv1d(
			in_channels=hidden_size,
			out_channels=hidden_size,
			kernel_size=kernel_size
		)
		self.conv_3 = nn.Conv1d(
			in_channels=hidden_size,
			out_channels=hidden_size,
			kernel_size=kernel_size
		)
		self.swish_1 = Swish()
		self.swish_2 = Swish()
		self.swish_3 = Swish()
		if norm_type == 'group':
			self.normalization_1 = nn.GroupNorm(
				num_groups=8,
				num_channels=hidden_size
			)
			self.normalization_2 = nn.GroupNorm(
				num_groups=8,
				num_channels=hidden_size
			)
			self.normalization_3 = nn.GroupNorm(
				num_groups=8,
				num_channels=hidden_size
			)
		else:
			self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
			self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
			self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
		
		self.pool = nn.MaxPool1d(kernel_size=2)
	
	def forward(self, input):
		conv1 = self.conv_1(input)
		x = self.normalization_1(conv1)
		x = self.swish_1(x)
		x = F.pad(x, pad=(self.kernel_size - 1, 0))
		
		x = self.conv_2(x)
		x = self.normalization_2(x)
		x = self.swish_2(x)
		x = F.pad(x, pad=(self.kernel_size - 1, 0))
		
		conv3 = self.conv_3(x)
		x = self.normalization_3(conv1 + conv3)
		x = self.swish_3(x)
		x = F.pad(x, pad=(self.kernel_size - 1, 0))
		
		x = self.pool(x)
		return x


class RNN(nn.Module):
	"""RNN module(cell type lstm or gru)"""
	
	def __init__(
			self,
			input_size,
			hid_size,
			num_rnn_layers=1,
			dropout_p=0.2,
			bidirectional=False,
			rnn_type='lstm',
	):
		super().__init__()
		
		if rnn_type == 'lstm':
			self.rnn_layer = nn.LSTM(
				input_size=input_size,
				hidden_size=hid_size,
				num_layers=num_rnn_layers,
				dropout=dropout_p if num_rnn_layers > 1 else 0,
				bidirectional=bidirectional,
				batch_first=True,
			)
		
		else:
			self.rnn_layer = nn.GRU(
				input_size=input_size,
				hidden_size=hid_size,
				num_layers=num_rnn_layers,
				dropout=dropout_p if num_rnn_layers > 1 else 0,
				bidirectional=bidirectional,
				batch_first=True,
			)
	
	def forward(self, input):
		outputs, hidden_states = self.rnn_layer(input)
		return outputs, hidden_states


class RNNModel(nn.Module):
	def __init__(
			self,
			input_size,
			hid_size,
			rnn_type,
			bidirectional,
			n_classes=1,
			kernel_size=5,
	):
		super().__init__()
		
		self.rnn_layer = RNN(
			input_size=hid_size,  # hid_size * 2 if bidirectional else hid_size,
			hid_size=hid_size,
			rnn_type=rnn_type,
			bidirectional=bidirectional
		)
		self.conv1 = ConvNormPool(
			input_size=input_size,
			hidden_size=hid_size,
			kernel_size=kernel_size,
		)
		self.conv2 = ConvNormPool(
			input_size=hid_size,
			hidden_size=hid_size,
			kernel_size=kernel_size,
		)
		self.avgpool = nn.AdaptiveAvgPool1d((1))
		self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)
	
	def forward(self, input):
		x = self.conv1(input)
		x = self.conv2(x)
		x = x.permute((0, 2, 1))
		x, _ = self.rnn_layer(x)
		x = x.permute((0, 2, 1))
		x = self.avgpool(x)
		x = x.view(-1, x.size(1) * x.size(2))
		x = F.sigmoid(self.fc(x))  # .squeeze(1)
		return x