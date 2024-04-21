import wandb
import train
import torch
import pprint
from config import ModelType, device, train_test_split, epochs

def run(config=None):
    with wandb.init(config=config, sync_tensorboard=True):
        config = wandb.config
        train.main(
            lr=config.lr, 
            optimizer=config.optimizer, 
            batch_size=config.batch_size ,
            epochs=epochs,
            train_test_split=train_test_split,
            device=device,
            modelType=ModelType.Regression,
            num_layers=config.num_layers,
            layer_width=config.layer_width
        )

def main():
    sweep_config = {
        'method': 'random'
    }

    parameters_dict = {
        'lr': {
            'values': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        },
        'optimizer': {
            'values': ["Adam", "SGD"]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256, 512]
        },
        'num_layers': {
            'values': [1, 2, 4, 6, 8]
        },
        'layer_width': {
            'values': [8, 16, 32, 64, 128]
        }
    }
    sweep_config['parameters'] = parameters_dict

    metric = {
        'name': 'valid/R2',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    
    pprint.pprint(sweep_config)
    
    sweep_id = wandb.sweep(sweep_config, project="SVAI")
    wandb.agent(sweep_id, run, count=10)

if __name__=='__main__':
    main()