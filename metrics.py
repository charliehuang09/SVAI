import config
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import cv2
import glob
from dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, r2_score
from dataset import Dataset
import torch
import pandas as pd
from model import Model
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import warnings
import imageio

warnings.filterwarnings("ignore")
        
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
    return r2_score(y_true=y_true, y_pred=y_pred)

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

def remap(model, index=0):
    dataset = Dataset("Train", 0.8, 'Regression', True, verbose=False)
    x_min = dataset.getxmin()
    x_max = dataset.getxmax()
    
    lightning = pd.read_pickle('cleanedData/lightning.pkl').iloc[index]
    population = pd.read_pickle('cleanedData/population.pkl').iloc[index]
    rain = pd.read_pickle('cleanedData/rain.pkl').iloc[index]
    biomass = pd.read_pickle('cleanedData/biomass.pkl').iloc[index]
    temperature = pd.read_pickle('cleanedData/temperature.pkl').iloc[index]
    humidity = pd.read_pickle('cleanedData/humidity.pkl').iloc[index]
    wind = pd.read_pickle('cleanedData/wind.pkl').iloc[index]
    soil_moisture = pd.read_pickle('cleanedData/soil_moisture.pkl').iloc[index]
    fireCCIL1982_2018 = pd.read_pickle('cleanedData/fireCCIL1982-2018.pkl').iloc[index]
    
    y = pd.read_pickle('cleanedData/fireCCIL1982-2018.pkl').iloc[index + 1]
    y = np.array(y)
    
    x = []
    x.append(fireCCIL1982_2018)
    x.append(lightning)
    x.append(population)
    x.append(rain)
    x.append(biomass)
    x.append(humidity)
    x.append(wind)
    x.append(soil_moisture)
    x.append(temperature)
    x = np.array(x, dtype=np.float32)
    
    y = y / np.nanmean(y)
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    map = np.zeros((50, 32))
    model.eval()
    for i in range(50):
        for j in range(32):
            if (torch.isnan(y[i][j]) or np.isnan(x[:, i, j]).any()):
                map[i][j] = np.nan
                y[i][j] = np.nan
                continue
            map[i][j] = model(x[:, i, j])[0]
    
    return map, y

def writeRemap(model, writer):
    matplotlib.use('agg')
    predsTrain = []
    ground_truth_Train = []
    
    predsValid = []
    ground_truth_Valid = []
    for i in trange(119):
        map, y = remap(model, i)
        
        fig, ax = plt.subplots(1, 2)
    
        ax[0].imshow(map)
        ax[0].set_title("Predictions")
        
        ax[1].imshow(y)
        ax[1].set_title("Ground Truth")
        
        if (i <= config.train_test_split * 120):
            predsTrain.append(map)
            ground_truth_Train.append(y)
            writer.add_figure("train/Remap", fig, i)
        else:
            predsValid.append(map)
            ground_truth_Valid.append(y)
            writer.add_figure("valid/Remap", fig, i)
        
    predsTrain = np.array(predsTrain).mean(axis=0)
    ground_truth_Train = np.array(ground_truth_Train).mean(axis=0)
    
    predsValid = np.array(predsValid).mean(axis=0)
    ground_truth_Valid = np.array(ground_truth_Valid).mean(axis=0)
    
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(predsTrain)
    ax[0].set_title("Predictions")
    
    ax[1].imshow(ground_truth_Train)
    ax[1].set_title("Ground Truth")
    
    writer.add_figure("train/RemapAverage", fig)
    
    
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(predsValid)
    ax[0].set_title("Predictions")
    
    ax[1].imshow(ground_truth_Valid)
    ax[1].set_title("Ground Truth")
    
    writer.add_figure("valid/RemapAverage", fig)