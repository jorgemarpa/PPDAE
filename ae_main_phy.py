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
from src.ae_training_phy import Trainer
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
                    help='were to is running (local, colab, gradient, [exalearn])')

parser.add_argument('--data', dest='data', type=str, default='PPD',
                    help='data used for training (MNIST, [PPD])')
parser.add_argument('--img-norm', dest='img_norm', type=str, default='global',
                    help='type of normalization for images ([global], image, None)')
parser.add_argument('--par-norm', dest='par_norm', type=str, default='T',
                    help='physical parameters are 0-1 scaled ([T],F)')
parser.add_argument('--subset', dest='subset', type=str, default='',
                    help='data subset ([],fexp1, 8702022)')

parser.add_argument('--optim', dest='optim', type=str, default='Adam',
                    help='Optimiazer ([Adam], SGD)')
parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp, cosine, plateau)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100,
                    help='total number of training epochs [100]')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stoping')

parser.add_argument('--cond', dest='cond', type=str, default='T',
                    help='physics conditioned AE ([F],T)')
parser.add_argument('--feed-phy', dest='feed_phy', type=str, default='F',
                    help='feed physics to decoder ([F],T)')
parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=32,
                    help='dimension of latent space [32]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for all layers [0.2]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3,
                    help='2D conv kernel size, encoder [3]')
parser.add_argument('--conv-blocks', dest='conv_blocks', type=int, default=5,
                    help='conv+actfx+pool blocks [5]')
parser.add_argument('--model-name', dest='model_name', type=str,
                    default='ConvLinTrans_AE', help='name of model')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project and save user defined flags
wandb.init(entity="deep_ppd", project="PPD-multiwave", tags=["multiwave"])
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed


# run main program
def run_code():
    # asses which device will be used, CPY or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Load Data #
    if args.data == 'PPD':
        dataset = ProtoPlanetaryDisks(machine=args.machine, transform=True,
                                      par_norm=str2bool(args.par_norm),
                                      subset=args.subset, image_norm=args.img_norm)
    elif args.data == 'MNIST':
        dataset = MNIST(args.machine)
    else:
        print('Error: Wrong dataset (MNIST, Proto Planetary Disk)...')
        raise

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
    # trainfeature, trainlabel = next(iter(train_loader))
    # print(f"Feature Batch Shape: {trainfeature.size()}")
    # print(f"Label Batch Shape: {trainlabel.size()}")

    print(f'Image dimension {dataset.img_dim}')
    print(f'Image channels {dataset.img_channels}')

    if args.data == 'PPD' and str2bool(args.cond):
        wandb.config.physics_dim = len(dataset.par_names)
    else:
        wandb.config.physics_dim = 0
        args.feed_phy = 'F'
    wandb.config.update(args, allow_val_change=True)

    print('Physic dimension: ', wandb.config.physics_dim)

    # Define AE model, Ops, and Train #
    # To used other AE models change the following line,
    # different types of AE models are stored in src/ae_model.py
    if args.model_name == 'ConvLinTrans_AE':
        model = ConvLinTrans_AE(latent_dim=args.latent_dim,
                                img_dim=dataset.img_dim,
                                dropout=args.dropout,
                                in_ch=dataset.img_channels,
                                kernel=args.kernel_size,
                                n_conv_blocks=args.conv_blocks,
                                phy_dim=wandb.config.physics_dim,
                                feed_phy=str2bool(args.feed_phy))
    elif args.model_name == 'ConvLin_AE':
        model = ConvLin_AE(latent_dim=args.latent_dim,
                           img_dim=dataset.img_dim,
                           dropout=args.dropout,
                           in_ch=dataset.img_channels,
                           kernel=args.kernel_size,
                           n_conv_blocks=args.conv_blocks,
                           phy_dim=wandb.config.physics_dim,
                           feed_phy=str2bool(args.feed_phy))
    elif args.model_name == 'ConvLinUpsample_AE':
        model = ConvLinUpsample_AE(latent_dim=args.latent_dim,
                                   img_dim=dataset.img_dim,
                                   dropout=args.dropout,
                                   in_ch=dataset.img_channels,
                                   kernel=args.kernel_size,
                                   n_conv_blocks=args.conv_blocks,
                                   phy_dim=wandb.config.physics_dim,
                                   feed_phy=str2bool(args.feed_phy))
    elif args.model_name == 'Forward_AE':
        model = Forward_AE(img_dim=dataset.img_dim,
                                   dropout=args.dropout,
                                   in_ch=dataset.img_channels,
                                   phy_dim=wandb.config.physics_dim)

    else:
        print('Wrong Model Name.')
        print('Please select model: ConvLinTrans_AE')
        sys.exit()

    # log model architecture and gradients to wandb
    wandb.watch(model, log='gradients')

    wandb.config.n_train_params = count_parameters(model)
    print('Summary:')
    print(model)
    print('Num of trainable params: ', wandb.config.n_train_params)
    print('\n')

    # Initialize optimizers
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        print('Error: please select a optimazer from Adam or SGD...')
        raise

    # Learning Rate scheduler
    if args.lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=25,
                                              gamma=0.5)
    elif args.lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif args.lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=25,
                                                         eta_min=1e-5)
    elif args.lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.1,
                                                         patience=10,
                                                         threshold=1e-4,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)

    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')

    # initialize trainer
    if args.model_name == 'Forward_AE':
      from src.ae_forward_trainer_phy import Trainer
      trainer = Trainer(model, optimizer, args.batch_size, wandb,
                      scheduler=scheduler, print_every=100,
                      device=device)
    else:
        from src.ae_training_phy import Trainer
        trainer = Trainer(model, optimizer, args.batch_size, wandb,
        scheduler=scheduler, print_every=100, device=device)

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
