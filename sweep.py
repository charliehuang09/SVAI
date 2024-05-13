import wandb
import train
import pprint
from config import device, modelType, train_test_split, shift

def run(config=None):
    with wandb.init(config=config, sync_tensorboard=True):
        config = wandb.config
        train.main(
            lr=config.lr, 
            optimizer=config.optimizer, 
            batch_size=config.batch_size ,
            epochs=config.epochs,
            num_layers=config.num_layers,
            layer_width=config.layer_width,
            dropout=config.dropout,
            
            device=device,
            train_test_split=train_test_split,
            modelType=modelType,
            shift=shift
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
            'values': ["SGD"]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256, 512]
        },
        'epochs': {
            'values': [3000]
        },
        'num_layers': {
            'values': [4, 6, 8, 16, 32]
        },
        'layer_width': {
            'values': [32, 64, 128]
        },
        'dropout': {
            'values': [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
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
    wandb.agent(sweep_id, run, count=25)

if __name__=='__main__':
    main()