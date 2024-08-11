import torch
from torch import nn
from typing import Literal


class Model(torch.nn.Module):
    def __init__(self, num_layers, layer_width, dropout,
                 modelType: Literal['Regression', 'Classification']):
        super().__init__()

        self.modelType: Literal['Regression', 'Classification'] = modelType
        self.input = nn.Linear(9, layer_width)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_width, layer_width))

        self.output = nn.Linear(layer_width, 9)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.output(x)

        if (self.modelType == 'Regression'):
            x = self.relu(x)
        if (self.modelType == 'Classification'):
            x = self.sigmoid(x)

        return x
