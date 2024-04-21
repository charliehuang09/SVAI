import config
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from model import RegressionModel
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, r2_score
import torch
import matplotlib
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
    matrix = matrix / matrix.sum()
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

def getF1Score(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    return f1_score(y_true, y_pred.detach().numpy(), average='weighted')

def getR2Score(y_pred, y_true):
    return r2_score(y_true, y_pred)

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
    matplotlib.use('agg')
    return plt.scatter(y_pred, y_true).get_figure()

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