import torch
from torch import optim
from dataset import Dataset
from model import Model
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary
from logger import Logger
from metrics import getConfusionMatrix, getAccuracy, getF1Score, getScatterPlot, getR2Score, writeRemap
from config import log_frequency
import warnings
from typing import Literal
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
def main(lr, optimizer, batch_size, epochs, train_test_split, device, modelType : Literal['Regression', 'Classification'], num_layers, layer_width, dropout, shift):
    torch.manual_seed(16312942289339198420)
    print(f"Seed: {torch.initial_seed()}")

    print(f"Model Type: {modelType}")

    model = Model(num_layers, layer_width, dropout, modelType)

    summary(model, (1, 8))
    
    model = model.to(device)

    train_dataset = Dataset("Train", train_test_split=train_test_split, modelType=modelType, shift=shift)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = Dataset("Valid", train_test_split=train_test_split, modelType=modelType, shift=shift)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    if (optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if (optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_fn = None
    writer = SummaryWriter()
    if (modelType == 'Regression'):
        loss_fn = torch.nn.MSELoss() #Regression

        trainR2Logger = Logger(writer, "train/R2")
        validR2Logger = Logger(writer, "valid/R2")

    if (modelType == 'Classification'):
        loss_fn = torch.nn.CrossEntropyLoss() #Classification

        trainAccuracyLogger = Logger(writer, "train/Accuracy")
        validAccuracyLogger = Logger(writer, "valid/Accuracy")

        trainF1Logger = Logger(writer, "train/F1")
        validF1Logger = Logger(writer, "valid/F1")

    trainLossLogger = Logger(writer, "train/Loss")
    validLossLogger = Logger(writer, "valid/Loss")

    for epoch in range(epochs):
        y_predTrain = []
        y_trueTrain = []
        model.train()
        for batch in train_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            if (epoch % log_frequency == 0):
                trainLossLogger.add(loss.item(), len(x))
                y_predTrain.extend(outputs[:, 0].tolist())
                y_trueTrain.extend(y[:, 0].tolist())
                
                if (modelType == 'Regression'):
                    pass
                if (modelType == 'Classification'):
                    trainAccuracyLogger.add(getAccuracy(outputs, y), 1)
                    trainF1Logger.add(getF1Score(outputs, y), 1)

        y_predValid = []
        y_trueValid = []
        model.eval()
        for batch in valid_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            if (epoch % log_frequency == 0):
                validLossLogger.add(loss.item(), len(x))
                y_predValid.extend(outputs[:, 0].tolist())
                y_trueValid.extend(y[:, 0].tolist())
                
                if (modelType == 'Regression'):
                    pass
                if (modelType == 'Classification'):
                    validAccuracyLogger.add(getAccuracy(outputs, y), 1)
                    validF1Logger.add(getF1Score(outputs, y), 1)
        if (epoch % log_frequency == 0):
            if (modelType == 'Regression'):
                print(f"Epoch: {epoch} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()} TrainR2: {getR2Score(y_predTrain, y_trueTrain)} ValidR2: {getR2Score(y_predValid, y_trueValid)}")

                trainR2Logger.write(getR2Score(y_predTrain, y_trueTrain))
                validR2Logger.write(getR2Score(y_predValid, y_trueValid))

            if (modelType == 'Classification'):
                print(f"Epoch: {epoch} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}Train Accuracy: {trainAccuracyLogger.get()} Valid Accuracy: {validAccuracyLogger.get()}")
                
                trainF1Logger.get()
                validF1Logger.get()
    
    if (modelType == 'Regression'):
        writer.add_figure("train/ScatterPlot", getScatterPlot(model, train_dataloader))
        writer.add_figure("valid/ScatterPlot", getScatterPlot(model, valid_dataloader))
        writeRemap(model, writer)
        writeRemap(model, writer)
    if (modelType == 'Classification'):
        writer.add_figure("train/ConfusionMatrix", getConfusionMatrix(model, train_dataloader))
        writer.add_figure("valid/ConfusionMatrix", getConfusionMatrix(model, valid_dataloader))
    
    writer.flush()
    
    torch.save(model, 'model.pt')

if __name__=='__main__':
    import config
    main(
        lr=config.lr, 
        optimizer=config.optimizer, 
        batch_size=config.batch_size ,
        epochs=config.epochs,
        num_layers=config.num_layers,
        layer_width=config.layer_width,
        dropout=config.dropout,
        
        device=config.device,
        train_test_split=config.train_test_split,
        modelType=config.modelType,
        shift=config.shift
        )