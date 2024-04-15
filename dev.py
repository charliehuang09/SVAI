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
        modelType=config.modelType
        )

if __name__=='__main__':
    main()
