import torch
from enum import Enum
ModelType = Enum('ModelType', ['Regression', 'Classification'], type=str)

lr = 0.001
optimizer = "Adam"
batch_size = 64
epochs = 10000
train_test_split=0.8
device = torch.device('cpu')
modelType = ModelType.Regression
num_layers = 3
layer_width = 8

log_frequency = 100