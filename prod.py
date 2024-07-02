import wandb
import train
import config
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    wandb.init(
    project="SVAI",
    sync_tensorboard=True,
    name=args.name,
    config={
        "lr": config.lr,
        "optimizer": config.optimizer,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "num_layers": config.num_layers,
        "layer_width": config.layer_width,
        "dropout": config.dropout,
        "train_test_split": config.train_test_split,
        "shift": config.shift
    },
    )
    train.main(
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
    
    wandb.save('model.pt')
    wandb.finish()

if __name__=='__main__':
    main()
