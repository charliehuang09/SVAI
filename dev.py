import train
import config
def main():
    train.main(
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

if __name__=='__main__':
    main()
