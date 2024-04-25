import torch
from typing import Literal

lr = 0.001
optimizer = "SGD"
batch_size = 1024
epochs = 5000
num_layers = 4
layer_width = 16
dropout = 0.05

device = torch.device('cpu')
modelType : Literal['Regression', 'Classification'] = 'Regression'
train_test_split=0.8

log_frequency = 100