import torch
from typing import Literal

lr = 0.001
optimizer = "SGD"
batch_size = 256
epochs = 10000
num_layers = 6
layer_width = 128
dropout = 0.1

device = torch.device('cpu')
modelType: Literal['Regression', 'Classification'] = 'Regression'
train_test_split = 0.8
shift = True

log_frequency = 10
