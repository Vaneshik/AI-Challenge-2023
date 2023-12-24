import torch

names = {
	'норма': 'normnal',
	'передний': 'front',
	'нижний': 'down',
	'перегородочный': 'septal',
	'передне-перегородочный': 'front_septal',
	'передне-боковой': 'front_down'
}

for k, v in names.items():
	model = torch.load(f'/Users/danil/AIIJC_FINAL/MODELS/Model_aug2_{k}')
	torch.save(model.state_dict(), f'/Users/danil/AIIJC_FINAL/inference/weights/convV2_{v}')
	
import torch

# model = torch.load('/Users/danil/AIIJC_FINAL/MODELS/Model_aug2_перегородочный')
# torch.save(model.state_dict(), f'/Users/danil/Downloads/test_save_dict')