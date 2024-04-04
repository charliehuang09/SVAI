import torch
from torch import nn

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(8, 24)

        self.fc1 = nn.Linear(24, 24)
        self.fc2 = nn.Linear(24, 24)

        self.output = nn.Linear(24, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.output(x)
        return x
class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(8, 24)

        self.fc1 = nn.Linear(24, 24)
        self.fc2 = nn.Linear(24, 24)

        self.output = nn.Linear(24, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.sigmoid(x)
        return x


def main():
    regressionModel = RegressionModel()
    classificationModel = ClassificationModel()

if __name__=='__main__':
    main()
