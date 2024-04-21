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
        "epochs": config.epochs,
        "optimizer": config.optimizer,
        "lr": config.lr,
        "train_test_split": config.train_test_split,
        "num_layers": config.num_layers,
        "layer_width": config.layer_width
    },
    )
    train.main(
        lr=config.lr, 
        optimizer=config.optimizer, 
        batch_size=config.batch_size ,
        epochs=config.epochs,
        train_test_split=config.train_test_split,
        device=config.device,
        modelType=config.modelType
        )
    wandb.finish()

if __name__=='__main__':
    main()
