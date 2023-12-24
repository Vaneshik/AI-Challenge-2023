import numpy as np
import tqdm
import torch
import pandas as pd
from train_all_signal import Dataset1D, smoothing
from os import listdir, path
import pickle
from tqdm.auto import tqdm
import scipy
from functions import dump_pickle
from MODELS.FullSignal.full_conv1d_front import Model1DPered
from MODELS.FullSignal.full_conv1d_pereg import Model1DPereg
from MODELS.FullSignal.full_conv1d_frontpereg import Model1DPeredPereg
from MODELS.FullSignal.full_conv1d_down import Model1DDown
from MODELS.FullSignal.full_conv1d_frontside import Model1DSideDown
from MODELS.FullSignal.conv_bilstm import CNN
from MODELS.FullSignal.se_resnet import Se_Resnet

with open('/Users/danil/AIIJC_FINAL/DATASETS/train_full_signals.pickle', mode='rb') as file:
	df_cleaned = pickle.load(file)
	
	
skipconcnn_dict = {
	'нижний': 64,
	'передний': 64,
	'перегородочный': 80,
	'передне-перегородочный': 70,
	'передне-боковой': 100
}
	
	
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

def predict(df: pd.DataFrame):
	device = 'mps'
	models_path = '/Users/danil/AIIJC_FINAL/MODELS'
	d = {
		'норма': list(range(0, 12)),
		'нижний': list(range(1, 11)),
		'передний': [7, 8, 9, 10],
		'перегородочный': list(range(1, 11)),
		'передне-перегородочный': [6, 7, 8, 9],
		'передне-боковой': [0, 1, 3, 9, 10, 11],
	}
	models = {
		'SCOCNN_перегородочный': (CNN(len(d['перегородочный']), hid_size=skipconcnn_dict['перегородочный']),
		                         'SkipConnectionCNN/septal/model_dict'),
		'SCOCNN_нижний': (CNN(len(d['нижний']), hid_size=skipconcnn_dict['нижний']),
		                 'SkipConnectionCNN/down/model_dict'),
		'SCOCNN_передний': (CNN(len(d['передний']), hid_size=skipconcnn_dict['передний']),
		                   'SkipConnectionCNN/front/model_dict'),
		'SCOCNN_передне-перегородочный': (CNN(len(d['передне-перегородочный']), hid_size=skipconcnn_dict['передне-перегородочный']),
		                                 'SkipConnectionCNN/front_septal/model_dict'),
		'SCOCNN_передне-боковой': (CNN(len(d['передне-боковой']), hid_size=skipconcnn_dict['передне-боковой']),
		                          'SkipConnectionCNN/front_down/model_dict'),
		'RESNET_перегородочный': (Se_Resnet(len(d['перегородочный'])), 'RESNET/septal/model_dict'),
		'RESNET_нижний': (Se_Resnet(len(d['нижний'])), 'RESNET/down/model_dict'),
		'RESNET_передний': (Se_Resnet(len(d['передний'])), 'RESNET/front/model_dict'),
		'RESNET_передне-перегородочный': (Se_Resnet(len(d['передне-перегородочный'])), 'RESNET/front_septal/model_dict'),
		'RESNET_передне-боковой': (Se_Resnet(len(d['передне-боковой'])), 'RESNET/front_down/model_dict'),
		'RESNET_норма': (Se_Resnet(len(d['норма'])), 'RESNET/normal/model_dict'),
	}
	
	df['signal'] = df['signal'].apply(lambda x: np.array([smoothing(i) for i in x]))
	
	signals = torch.Tensor(df.signal)
	names = df['record_name']
	ans_df = pd.DataFrame()
	
	for k, (model, state_dict_path) in models.items():
		print(k)
		class_ = k[7:]
		model.load_state_dict(torch.load(path.join(models_path, state_dict_path)))
		model = model.to(device)
		res = []
		res_proba = []
		for chunk_signal in torch.chunk(signals, len(signals)):
			s = chunk_signal[:, d[class_]]
			roll_s = torch.Tensor(np.array([np.roll(s, i, axis=1) for i in np.arange(0, 4000, 5000)])).to(device)
			roll_s = roll_s.squeeze(1)
			if k[0] == 'S':
				model.avgpool.register_forward_hook(get_activation('avgpool'))
				result = model(roll_s).cpu().detach().numpy()
				active = activation['avgpool'].detach().cpu().squeeze(2).numpy()
				
			else:
				model.linear1.register_forward_hook(get_activation('linear1'))
				result = model(roll_s).cpu().detach().numpy()
				active = activation['linear1'].detach().cpu().numpy()
			
			res_proba.append(result.mean(axis=0))
			res.append(active.mean(axis=0))
			
		# print(res[0].shape)
		# res = np.array(res)
		# print(res.shape)
		ans_df[k] = res
		ans_df[k + '_proba'] = res_proba
	ans_df['record_name'] = names
	
	return ans_df


ans = predict(df_cleaned)
dump_pickle('/Users/danil/AIIJC_FINAL/DATASETS/pred_train_full_RS.pickle', ans)