import numpy as np
import datetime
import torch
import torch.nn as nn
from src.utils import plot_recon_wall, plot_multi_recon_wall, plot_latent_space
from src.training_callbacks import EarlyStopping


class Trainer():
    """
    A training class for VAE model. Incorporate batch-training/testing loop,
    training callbacks, lerning rate scheduler, W&B logger with metrics reports and
    reconstruction plots.
    ...
    Attributes
    ----------
    train_loss  : dict
        dictionary with training metrics
    test_loss   : dict
         dictionary with testing metrics
    num_steps   : int
        number of training steps
    print_every : int
        report metric values every N training steps
    beta        : str
        value of beta hyperparameter that weight the KLD term in the loss function
    wb          : object
        weight and biases logger
    Methods
    -------
    _loss(self, x, xhat, mu, logvar, train=True, ep=0)
        calculate loss metrics and add them to logger
    _train_epoch(self, data_loader, epoch)
        do model training in a given epoch
    _test_epoch(self, test_loader, epoch)
        do model testing in a given epoch
    _report_train(self, i)
        report training metrics to W&B logger and standard terminal
    _report_test(self, ep)
        report test metrics to W&B logger and standard terminal
    train(self, train_loader, test_loader, epochs, data_ex,
          machine='local', save=True, early_stop=False)
        complete epoch training/validation/report loop
    """
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, print_every=50,
                 device='cpu'):
        """
        Parameters
        ----------
        model       : pytorch module
            pytorch VAE model
        optimizer   : pytorch optimizer
            pytorch optimizer object
        batch_size  : int
            size of batch for training loop
        wandb       : object
            weight and biases logger
        scheduler   : object
            pytorch learning rate schduler, default is None
        print_every : int
            step interval for report printing
        device      : str
            device where model will run (cpu, gpu)
        """
        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ', next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {'Loss': []}
        self.test_loss = {'Loss': []}
        self.num_steps = 0
        self.print_every = print_every
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.wb = wandb


    def _loss(self, x, xhat, train=True, ep=0):
        """Evaluates loss function and add reports to the logger.

        Parameters
        ----------
        x      : tensor
            tensor of real values
        xhat   : tensor
            tensor of predicted values
        train  : bool
            wheather is training step or not
        ep     : int
            epoch value of training loop
        Returns
        -------
        loss
            loss value
        """
        loss = self.bce_loss(xhat, x)

        if train:
            self.train_loss['Loss'].append(loss.item())
        else:
            self.test_loss['Loss'].append(loss.item())

        return loss

    def _train_epoch(self, data_loader, epoch):
        """Training loop for a given epoch. Triningo goes over
        batches, images and latent space plots are logged to
        W&B
        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
        Returns
        -------
        """
        # switch model to training mode
        self.model.train()
        # iterate over len(data)/batch_size
        z_all = []
        xhat_plot, x_plot = [], []
        for i, (img, phy) in enumerate(data_loader):
            self.num_steps += 1
            self.opt.zero_grad()
            img = img.to(self.device)
            phy = phy.to(self.device)

            xhat, z = self.model(img, phy=phy)
            # calculate loss value
            loss = self._loss(img, xhat, train=True, ep=epoch)
            # calculate the gradients
            loss.backward()
            # perform optimization step accordig to the gradients
            self.opt.step()

            self._report_train(i)
            # aux variables for latter plots
            z_all.append(z.data.cpu().numpy())
            if i == len(data_loader) - 2:
                xhat_plot = xhat.data.cpu().numpy()
                x_plot = img.data.cpu().numpy()
            #if i == 1:
            #    print('WARNING: using only 1st batch')
            #    break

        z_all = np.concatenate(z_all)
        z_all = z_all[np.random.choice(z_all.shape[0], 5000,
                                       replace=False), :]
        print(z_all.shape)

        # plot reconstructed images ever 2 epochs
        if epoch % 2 == 0:
            nchnnels = data_loader.img_channels
            if nchnnels > 1:
                wall = plot_multi_recon_wall(xhat_plot, x_plot, epoch=epoch, log=True)
                
            else: wall = plot_recon_wall(xhat_plot, x_plot, epoch=epoch, log=True)
            self.wb.log({'Train_Recon':  self.wb.Image(wall)},
                        step=self.num_steps)

        if epoch % 2 == 0:
            latent_plot = plot_latent_space(z_all, y=None)
            self.wb.log({'Latent_space': self.wb.Image(latent_plot)},
                        step=self.num_steps)

    def _test_epoch(self, test_loader, epoch):
        """Testing loop for a given epoch. Triningo goes over
        batches, images and latent space plots are logged to
        W&B logger
        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
        Returns
        -------
        """
        # swich model to evaluation mode, this make it deterministic
        self.model.eval()
        with torch.no_grad():
            xhat_plot, x_plot = [], []

            for i, (img, phy) in enumerate(test_loader):
                # send data to current device
                img = img.to(self.device)
                phy = phy.to(self.device)
                xhat, z = self.model(img, phy=phy)

                # calculate loss value
                loss = self._loss(img, xhat, train=False, ep=epoch)

                # aux variables for plots
                if i == len(test_loader) - 2:
                    xhat_plot = xhat.data.cpu().numpy()
                    x_plot = img.data.cpu().numpy()

        self._report_test(epoch)

        # plot reconstructed images ever 2 epochs
        if epoch % 2 == 0:
            nchnnels = data_loader.img_channels
            if nchnnels > 1:
                wall = plot_multi_recon_wall(xhat_plot, x_plot, epoch=epoch, log=True)
            else: wall = plot_recon_wall(xhat_plot, x_plot, epoch=epoch, log=True)
            self.wb.log({'Test_Recon':  self.wb.Image(wall)},
                        step=self.num_steps)

        return loss

    def train(self, train_loader, test_loader, epochs,
              save=True, early_stop=False):
        """Full training loop over all epochs. Model is saved after
        training is finished.
        Parameters
        ----------
        train_loader : pytorch object
            data loader object with training items
        test_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number
        save        : bool
            wheather to save or not final model
        early_stop  : bool
            whaether to use early stop callback which stops training
            when validation loss converges
        Returns
        -------
        """

        # hold samples, real and generated, for initial plotting
        if early_stop:
            early_stopping = EarlyStopping(patience=10, min_delta=.1,
                                           verbose=True)

        # train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss = self._test_epoch(test_loader, epoch)

            # update learning rate according to cheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss)
                else:
                    self.sch.step()

            # report elapsed time per epoch and total run tume
            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start
            print('Time per epoch: ', epoch_time.seconds, ' s')
            print('Elapsed time  : %.2f m' % (elap_time.seconds/60))
            print('##'*20)

            # early stopping
            if early_stop:
                early_stopping(val_loss.cpu())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(), '%s/model.pt' %
                       (self.wb.run.dir))

    def _report_train(self, i):
        """Report training metrics to logger and standard output.
        Parameters
        ----------
        i : int
            training step
        Returns
        -------
        """
        # ------------------------ Reports ---------------------------- #
        # print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0:
            print("Training iteration %i, global step %i" %
                  (i + 1, self.num_steps))
            print("Loss: %.6f" % (self.train_loss['Loss'][-1]))

            self.wb.log({'Train_Loss': self.train_loss['Loss'][-1]},
                        step=self.num_steps)
            print("__"*20)

    def _report_test(self, ep):
        """Report testing metrics to logger and standard output.
        Parameters
        ----------
        i : int
            testing step
        Returns
        -------
        """
        # ------------------------ Reports ---------------------------- #
        # print scalars to std output and save scalars/hist to W&B
        print('*** TEST LOSS ***')
        print("Epoch %i, global step %i" % (ep, self.num_steps))
        print("Loss: %.6f" % (self.test_loss['Loss'][-1]))

        self.wb.log({'Test_Loss': self.test_loss['Loss'][-1]},
                    step=self.num_steps)
        print("__"*20)
