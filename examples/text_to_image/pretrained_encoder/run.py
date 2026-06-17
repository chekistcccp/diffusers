import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

#from model import DAN

#comment off and replace with trainer_deepset, 10/13/2025
from trainer import birdClassifier
#from trainer_deepset import birdClassifier

#from linformer_trainer import birdClassifier
from dataset import speciesData

torch.set_float32_matmul_precision('medium')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train(args):
    with open(args.config, 'rb') as f:
        config = yaml.safe_load(f)

    name = f"{config['name']}_{args.dataset}"
    logger = pl_loggers.TensorBoardLogger("logs", name=name)

    model = birdClassifier(**config)
    print('>>>>> created model')
    
    """ 
    train_dl, valid_dl,_ = species_dataloader(
        train_path = config['traindata'],
        vaid_path = config['data_path'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
    )
    """
    
    train_ds = speciesData(config['train_data'],train=True )
    valid_ds = speciesData(config['valid_data'],valid=True )
    print(">>>>> build Dataset: train_ds and valid_ds")

    train_dl = DataLoader(train_ds, shuffle=True, batch_size = config['batch_size'], num_workers = config['num_workers'], drop_last = True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size = config['batch_size'], num_workers = config['num_workers'], drop_last = True)
    print(">>>>> build DataLoader: train_dl and valid_dl")

    if config['classify']:
        early_stop = pl.callbacks.EarlyStopping(
            monitor = 'val_acc', patience=3, mode='max'
        )

        checkpoint = pl.callbacks.ModelCheckpoint(dirpath='logs/ckpts-16birds-DAN/', save_last='link')
        trainer = pl.Trainer(
            max_steps=config['steps'],
            accelerator="gpu",
            devices=config['gpus'], 
            val_check_interval=0.25,
            accumulate_grad_batches=config['accumulate_grad_batches'],
            precision = config['precision'],
            callbacks=[checkpoint],
            logger=logger,
        )
    else:
        # pretraining
        checkpoint=pl.callbacks.ModelCheckpoint(monitor='val_loss')
        trainer = pl.Trainer(
            max_steps = config['steps'],
            accelerator="gpu",
            devices=config['gpus'], 
            precision = config['precision'],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    trainer.fit(model, train_dl, valid_dl)


def test(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    trainer = pl.Trainer()
    test_ds = speciesData(args.dataset, test=True)
    test_dl = DataLoader(test_ds, shuffle=True, batch_size = config['batch_size'], num_workers = config['num_workers'], drop_last = True)
 
    model = birdClassifier.load_from_checkpoint(args.checkpoint)
    birdClassifier.mode = "test"
    trainer.test(model, test_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="")
    
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("--config", type=str)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("--checkpoint", type=str)
    parser_test.add_argument("--config", type=str)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)

