import torch
from modelType import ModelType
#BaseLine 
lr = 0.001
optimizer = torch.optim.Adam
batch_size = 64
epochs = 100
train_test_split=0.8
device = torch.device('cpu')
modelType = ModelType.Regression

