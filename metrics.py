import config
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from model import RegressionModel
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch
import matplotlib.pyplot as plt
def getConfusionMatrix(model, dataloader):
    device = config.device
    y_true = []
    y_pred = []
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x).flatten()
        outputs[outputs <= 0.5] = 0
        outputs[outputs > 0.5] = 1
        y[y <= 0.5] = 0
        y[y > 0.5] = 1
        y_true.extend(y.tolist())
        y_pred.extend(outputs.tolist())
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    matrix = heatmap(matrix, annot=True, fmt='.3g')
    matrix = matrix.get_figure()
    return matrix

def getAccuracy(y_pred, y):
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    correct = torch.sum(y_pred == y).item()
    wrong = torch.sum(y_pred != y).item()
    return correct / (correct + wrong)

def getF1Score(y_pred, y):
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    return f1_score(y, y_pred.detach().numpy(), average='weighted')

def getScatterPlot(model, dataloader):
    device = config.device
    y_true = []
    y_pred = []
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x).flatten()
        y_true.extend(y.tolist())
        y_pred.extend(outputs.tolist())
    return plt.scatter(y_pred, y_true).get_figure()

def getR2Score(model, dataloader):
    device = config.device
    y_true = []
    y_pred = []
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x).flatten()
        y_true.extend(y.tolist())
        y_pred.extend(outputs.tolist())
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    r2Score = R2Score()
    r2Score.update(y_pred, y_true)
    return r2Score.compute()



def main():
    model = RegressionModel()
    dataset = Dataset("Train")
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    getConfusionMatrix(model, dataloader)

    x, y = next(iter(dataloader))
    print(getAccuracy(model(x), y))

    print(getF1Score(model(x), y))
    

if __name__=='__main__':
    main()