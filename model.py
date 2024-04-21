import torch
from torch import nn
import config
from torchsummary import summary
class RegressionModel(torch.nn.Module):
    def __init__(self, num_layers, layer_width):
        super().__init__()

        self.input = nn.Linear(8, layer_width)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_width, layer_width))

        self.output = nn.Linear(layer_width, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            
        x = self.output(x)
        x = self.relu(x)
        return x
    
class ClassificationModel(torch.nn.Module):
    def __init__(self, num_layers, layer_width):
        super().__init__()

        self.input = nn.Linear(8, layer_width)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_width, layer_width))

        self.output = nn.Linear(layer_width, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)

        x = self.output(x)
        x = self.sigmoid(x)
        return x


def main():
    regressionModel = RegressionModel(config.num_layers, config.layer_width)
    classificationModel = ClassificationModel()
    
    summary(regressionModel, (1, 8))

if __name__=='__main__':
    main()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):

#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(Net, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         current_dim = input_dim
#         self.layers = nn.ModuleList()
#         for hdim in hidden_dim:
#             self.layers.append(nn.Linear(current_dim, hdim))
#             current_dim = hdim
#         self.layers.append(nn.Linear(current_dim, output_dim))

#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x))
#         out = F.softmax(self.layers[-1](x))
#         return out    