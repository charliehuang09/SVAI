import torch
from dataset import TrainDataset, ValidDataset
from model import Model
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary
from logger import Logger
import wandb
import config
from torch.utils.tensorboard import SummaryWriter

def main():
    model = Model()

    summary(model, (1, 7))
    
    device = config.device
    model = model.to(device)

    train_dataset = TrainDataset(config.train_test_split)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

    valid_dataset = ValidDataset(config.train_test_split)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)

    optimizer = config.optimizer(model.parameters(), lr=config.lr)

    loss_fn = torch.nn.MSELoss() # Regression
    # loss_fn = torch.nn.CrossEntropyLoss() # Classification

    writer = SummaryWriter()
    trainLossLogger = Logger(writer, "train/LossLogger")
    validLossLogger = Logger(writer, "valid/LossLogger")

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

        for batch in valid_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)

            validLossLogger.add(loss.item(), len(x))

        print(f"Epoch: {epoch + 1} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}")

if __name__=='__main__':
    main()
