"""AE main training script
This script allows the user to train an AE model using Protoplanetary disk
images loaded with the 'dataset.py' class, AE model located in 'ae_model.py' class,
and the trainig loop coded in 'ae_training.py'. 
The script also uses Weight & Biases framework to log metrics, model hyperparameters, configuration parameters, and training figures.
This file contains the following
functions:
    * run_code - runs the main code
For help, run:
    python ae_main.py --help
"""

import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.dataset_large import ProtoPlanetaryDisks
from src.ae_model_phy import *
from src.ae_forward_trainer_phy import Trainer
from src.utils import count_parameters, str2bool
import wandb

torch.autograd.set_detect_anomaly(True)

# set random seed
rnd_seed = 13
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
torch.cuda.manual_seed_all(rnd_seed)
#os.environ['PYTHONHASHSEED'] = str(rnd_seed)

# program flags
parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--machine', dest='machine', type=str, default='local',
                    help='were to is running (local, colab, [exalearn])')
#erased latent dimension option, conv_blocks, data (args.data), args.optim, Kernel_size, lr_sch
parser.add_argument('--img-norm', dest='img_norm', type=str, default='global',
                    help='type of normalization for images ([global], image)')
parser.add_argument('--par-norm', dest='par_norm', type=str, default='T',
                    help='physical parameters are 0-1 scaled ([T],F)')
parser.add_argument('--subset', dest='subset', type=str, default='25052021',
                    help='data subset ([25052021],fexp1)')
parser.add_argument('--part-num', dest='par_num', type=int, 
                    default=1, help='partition subset number ([1],2,3,4,5)')
parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100,
                    help='total number of training epochs [100]')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stoping')
parser.add_argument('--transform', dest='transform', type=bool, default=True,
                    help='applies transformation to images ([True], False)')
parser.add_argument('--cond', dest='cond', type=str, default='T',
                    help='physics conditioned AE ([F],T)')
parser.add_argument('--feed-phy', dest='feed_phy', type=str, default='T',
                    help='feed physics to decoder (F,[T)]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for all layers [0.2]')
parser.add_argument('--model-name', dest='model_name', type=str, 
                    default='Forward_AE', help='name of model ([Forward_AE], Dev_Forward_AE)')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments for runtime labels')
args = parser.parse_args()

# Initialize W&B project and save user defined flags
wandb.init(entity="deep_ppd", project="PPD-forward", tags=["forward"])
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed


# run main program
def run_code():
    # asses which device will be used, CPY or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Load Data #
    dataset = ProtoPlanetaryDisks(machine=args.machine, transform=args.transform,
                                  par_norm=str2bool(args.par_norm),
                                  subset=args.subset, image_norm=args.img_norm, par_num=args.par_num)

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()
    print('Dataset size: ', len(dataset))
    
    # data loaders for training and testing
    train_loader, val_loader, _ = dataset.get_dataloader(batch_size=args.batch_size,
                                                         shuffle=True,
                                                         val_split=.2,
                                                         random_seed=rnd_seed)
      
    wandb.config.physics_dim = len(dataset.par_names)
    wandb.config.update(args, allow_val_change=True)

    print('Physic dimension: ', wandb.config.physics_dim)

    # Define AE model, Ops, and Train #
    # To used other AE models change the following line,
    # different types of AE models are stored in src/ae_model_phy.py
    if args.model_name == 'Forward_AE':
        model = Forward_AE(img_dim=dataset.img_dim,
                                   dropout=args.dropout,
                                   in_ch=dataset.img_channels,
                                   phy_dim=wandb.config.physics_dim)
    elif args.model_name == 'Dev_Forward_AE':
        model = Dev_Forward_AE(img_dim=dataset.img_dim,
                                   dropout=args.dropout,
                                   in_ch=dataset.img_channels,
                                   phy_dim=wandb.config.physics_dim)
        
    else:
        print('Wrong Model Name.')
        print('Please select model: Forward_AE or Dev_Forward_AE')
        sys.exit()
    
    # log model architecture and gradients to wandb
    wandb.watch(model, log='gradients')

    wandb.config.n_train_params = count_parameters(model)
    print('Summary:')
    print(model)
    print('Num of trainable params: ', wandb.config.n_train_params)
    print('\n')

    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    print('Optimizer    :', optimizer)

    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')
    
    # initialize trainer
    trainer = Trainer(model, optimizer, args.batch_size, wandb,
                      print_every=100, device=device)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    # run training/testing iterations
    trainer.train(train_loader, val_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)


if __name__ == "__main__":
    print('Running in: ', args.machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
