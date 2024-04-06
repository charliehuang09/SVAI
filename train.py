import torch
from dataset import Dataset
from model import RegressionModel, ClassificationModel
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary
from logger import Logger
import config
from modelType import ModelType
from metrics import getConfusionMatrix, getAccuracy, getF1Score, getScatterPlot
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
def main():

    print(f"Model Type: {config.modelType}")

    if (config.modelType == ModelType.Regression):
        model = RegressionModel()
    if (config.modelType == ModelType.Classification):
        model = ClassificationModel()

    summary(model, (1, 8))
    
    device = config.device
    model = model.to(device)

    train_dataset = Dataset("Train")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

    valid_dataset = Dataset("Valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)

    optimizer = config.optimizer(model.parameters(), lr=config.lr)

    loss_fn = None
    writer = SummaryWriter()
    if (config.modelType == ModelType.Regression):
        loss_fn = torch.nn.MSELoss() #Regression

    if (config.modelType == ModelType.Classification):
        loss_fn = torch.nn.CrossEntropyLoss() #Classification

        trainAccuracyLogger = Logger(writer, "train/Accuracy")
        validAccuracyLogger = Logger(writer, "valid/Accuracy")

        trainF1Logger = Logger(writer, "train/F1")
        validF1Logger = Logger(writer, "valid/F1")

    trainLossLogger = Logger(writer, "train/Loss")
    validLossLogger = Logger(writer, "valid/Loss")

    for epoch in range(config.epochs):
        for batch in train_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)
            loss.backward()
            optimizer.step()

            trainLossLogger.add(loss.item(), len(x))

            if (config.modelType == ModelType.Regression):
                pass
            if (config.modelType == ModelType.Classification):
                trainAccuracyLogger.add(getAccuracy(outputs, y), 1)
                trainF1Logger.add(getF1Score(outputs, y), 1)

        for batch in valid_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)

            validLossLogger.add(loss.item(), len(x))

            if (config.modelType == ModelType.Regression):
                pass
            if (config.modelType == ModelType.Classification):
                validAccuracyLogger.add(getAccuracy(outputs, y), 1)
                validF1Logger.add(getF1Score(outputs, y), 1)

        if (config.modelType == ModelType.Regression):
            print(f"Epoch: {epoch + 1} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}")

        if (config.modelType == ModelType.Classification):
            print(f"Epoch: {epoch + 1} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}Train Accuracy: {trainAccuracyLogger.get()} Valid Accuracy: {validAccuracyLogger.get()}")
            
            trainF1Logger.get()
            validF1Logger.get()
    
    if (config.modelType == ModelType.Regression):
        writer.add_figure("train/ScatterPlot", getScatterPlot(model, train_dataloader))
    if (config.modelType == ModelType.Classification):
        writer.add_figure("train/ConfusionMatrix", getConfusionMatrix(model, train_dataloader))
        writer.add_figure("valid/ConfusionMatrix", getConfusionMatrix(model, valid_dataloader))
    
    writer.flush()

if __name__=='__main__':
    main()
