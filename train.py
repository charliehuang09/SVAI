import torch
from dataset import Dataset
from model import Model
from tqdm import trange
from torch.utils.data import DataLoader
from torchsummary import summary

def main():
    model = Model()

    summary(model, (1, 7))

    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in trange(30):
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs[:, 0], y)
            loss.backward()
            optimizer.step()

if __name__=='__main__':
    main()
