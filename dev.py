import train
import config
def main():
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

if __name__=='__main__':
    main()
