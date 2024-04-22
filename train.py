import torch
from torch import optim
from dataset import Dataset
from model import RegressionModel, ClassificationModel
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary
from logger import Logger
from metrics import getConfusionMatrix, getAccuracy, getF1Score, getScatterPlot, getR2Score
from config import ModelType, log_frequency
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
def main(lr, optimizer, batch_size, epochs, train_test_split, device, modelType, num_layers, layer_width):
    torch.manual_seed(16312942289339198420)
    print(f"Seed: {torch.initial_seed()}")

    print(f"Model Type: {modelType}")

    if (modelType == ModelType.Regression):
        model = RegressionModel(num_layers, layer_width)
    if (modelType == ModelType.Classification):
        model = ClassificationModel()

    summary(model, (1, 8))
    
    model = model.to(device)

    train_dataset = Dataset("Train", train_test_split, modelType)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = Dataset("Valid", train_test_split, modelType)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    if (optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if (optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_fn = None
    writer = SummaryWriter()
    if (modelType == ModelType.Regression):
        loss_fn = torch.nn.MSELoss() #Regression

        trainR2Logger = Logger(writer, "train/R2")
        validR2Logger = Logger(writer, "valid/R2")

    if (modelType == ModelType.Classification):
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

            y_predTrain.extend(outputs.tolist())
            y_trueTrain.extend(y.tolist())

            if (epoch % log_frequency == 0):
                if (modelType == ModelType.Regression):
                    pass
                if (modelType == ModelType.Classification):
                    trainAccuracyLogger.add(getAccuracy(outputs, y), 1)
                    trainF1Logger.add(getF1Score(outputs, y), 1)

        y_predValid = []
        y_trueValid = []
        for batch in valid_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)

            validLossLogger.add(loss.item(), len(x))

            y_predValid.extend(outputs.tolist())
            y_trueValid.extend(y.tolist())
            
            if (epoch % log_frequency == 0):
                if (modelType == ModelType.Regression):
                    pass
                if (modelType == ModelType.Classification):
                    validAccuracyLogger.add(getAccuracy(outputs, y), 1)
                    validF1Logger.add(getF1Score(outputs, y), 1)
        if (epoch % log_frequency == 0):
            if (modelType == ModelType.Regression):
                print(f"Epoch: {epoch} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()} TrainR2: {getR2Score(y_predTrain, y_trueTrain)} ValidR2: {getR2Score(y_predValid, y_trueValid)}")

                trainR2Logger.write(getR2Score(y_predTrain, y_trueTrain))
                validR2Logger.write(getR2Score(y_predValid, y_trueValid))

            if (modelType == ModelType.Classification):
                print(f"Epoch: {epoch} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}Train Accuracy: {trainAccuracyLogger.get()} Valid Accuracy: {validAccuracyLogger.get()}")
                
                trainF1Logger.get()
                validF1Logger.get()
    
    if (modelType == ModelType.Regression):
        writer.add_figure("train/ScatterPlot", getScatterPlot(model, train_dataloader))
        writer.add_figure("valid/ScatterPlot", getScatterPlot(model, valid_dataloader))
    if (modelType == ModelType.Classification):
        writer.add_figure("train/ConfusionMatrix", getConfusionMatrix(model, train_dataloader))
        writer.add_figure("valid/ConfusionMatrix", getConfusionMatrix(model, valid_dataloader))
    
    writer.flush()

if __name__=='__main__':
    import config
    main(
        lr=config.lr, 
        optimizer=config.optimizer, 
        batch_size=config.batch_size ,
        epochs=config.epochs,
        train_test_split=config.train_test_split,
        device=config.device,
        modelType=config.modelType,
        num_layers=config.num_layers,
        layer_width=config.layer_width
        )