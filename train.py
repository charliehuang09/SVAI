import torch
from dataset import TrainDataset, ValidDataset
from model import Model
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

def main():
    model = Model()

    summary(model, (1, 7))

    train_dataset = TrainDataset(0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    valid_dataset = ValidDataset(0.8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    device = torch.device('mps')

    writer = SummaryWriter()
    trainLossLogger = Logger(writer, "train/LossLogger")
    validLossLogger = Logger(writer, "valid/LossLogger")

    for epoch in range(100):
        for batch in train_dataloader:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)
            loss.backward()
            optimizer.step()

            trainLossLogger.add(loss.item(), 64)

        for batch in valid_dataloader:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)
            loss.backward()
            optimizer.step()

            validLossLogger.add(loss.item(), 64)

        print(f"Epoch: {epoch + 1} Train Loss: {trainLossLogger.get()} Valid Loss: {validLossLogger.get()}")

if __name__=='__main__':
    main()
