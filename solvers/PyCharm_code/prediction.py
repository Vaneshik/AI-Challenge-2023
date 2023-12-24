import numpy as np
import tqdm
import torch
import pandas as pd
from main import MyDataset
from os import listdir, path
import pickle
from tqdm.auto import tqdm
from MODELS.lead_conv2d import LeadModel

with open('/Users/danil/AIIJS_PROJECT/clean_datasets/clean_test.pickle', mode='rb') as file:
	df_cleaned = pickle.load(file)
	
torch.PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
	

def predict(df: pd.DataFrame):
	dataset = MyDataset(df, is_train=False)
	device = 'mps'
	models_path = '/Users/danil/AIIJC_FINAL/MODELS'
	
	predictions = []
	signals = torch.Tensor(dataset.signals).to(device).unsqueeze(1)
	names = dataset.labels
	ans_df = pd.DataFrame()
	for name in listdir(models_path):
		if 'Model' in name:
			model = torch.load(path.join(models_path, name))
			model = model.to(device)
			res = []
			for chunk_signal in tqdm(torch.chunk(signals, len(signals))):
				res.append(model(chunk_signal).cpu().detach().numpy())
			res = np.concatenate(res)
			ans_df[name] = list(res)
	ans_df['record_name'] = names
		
	return ans_df


ans = predict(df_cleaned)
ans.to_csv('/Users/danil/AIIJC_FINAL/DATASETS/pred_test.csv')