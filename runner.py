import os
import torch
from tqdm import tqdm
from logger import Logger, AverageMeter


class Runner:

    def fit(self):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

    def to(self, device=None, dtype=torch.float32):
        """
        set default torch device and dtype
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dtype = dtype
        self.device = device


class RunnerExample(Runner):

    def __init__(self,
                 model,  # models
                 criterion,  # criterion
                 optimizer,  # optimizer for the parameters
                 train_loader, test_loader,  # data
                 total_epochs, start_epoch=0,  # training epochs
                 scheduler=None,  # lr scheduler
                 logpath=None,  # logger info
                 dtype=torch.float32,  # tensor dtpye
                 device=None,  # device to be used
                 resume_path=None,  # resume the model from a given checkpoint
                 ):

        ## set torch device and dtype
        super(RunnerExample, self).to(device, dtype)

        ## models
        self.model = model.to(self.device, self.dtype)

        ## criterions
        self.criterion = criterion
        self.best_metric = 9999

        ## optimizers
        self.optimizer = optimizer

        ## learning rate scheduler
        self.scheduler = scheduler

        ## data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        ## epochs for the training
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch

        ## set the logger path
        self.log = Logger(logpath)

        ## optionally resume training from a checkpoint
        if resume_path is not None:
            if os.path.isfile(resume_path):
                tqdm.write("=> loading checkpoint '{}'".format(resume_path))

                ## lead checkpoint
                checkpoint = torch.load(resume_path)

                ## get checkpoint state
                self.start_epoch = checkpoint['epoch']
                self.best_metric = checkpoint['best_metric']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

                tqdm.write("=> loaded checkpoint '{}' (epoch {})"
                           .format(resume_path, checkpoint['epoch']))
            else:
                tqdm.write("=> no checkpoint found at '{}'".format(resume_path))

        ## log transforms
        self.log.log_transforms(self.train_loader, self.test_loader)

    def fit(self, plot_iter_train=-1, plot_iter_test=-1):
        """
        Perform the whole training process with the information given at initialization time.
        """

        current_epoch = self.start_epoch
        while current_epoch < self.total_epochs:

            ## step the lr schedulers
            if self.scheduler is not None: self.scheduler.step()

            self.do_iteration(is_train=True, current_epoch=current_epoch, plot_iter=plot_iter_train)
            current_metric = self.do_iteration(is_train=False, current_epoch=current_epoch, plot_iter=plot_iter_test)

            is_best = current_metric < self.best_metric
            self.best_metric = min(current_metric, self.best_metric)

            self.log.save_model({'epoch': current_epoch,
                                 'state_dict': self.model.state_dict(),
                                 'best_metric': self.best_metric,
                                 'optimizer': self.optimizer.state_dict(), },
                                is_best, epoch=current_epoch)

            current_epoch += 1

    def do_iteration(self, is_train, current_epoch=-1):
        """
        Perform one epoch using the information given at initialization time.
        """
        ## set model in the proper mode and select loader
        if is_train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.test_loader

        ## create object to log progress
        loss_meter = AverageMeter()

        ## set progress bar
        pr_bar = tqdm(loader, total=len(loader))

        ## start loop
        with torch.set_grad_enabled(is_train):
            for iter, (inputs, outputs) in enumerate(pr_bar):

                ## get data and move to the correct type and device
                inputs = inputs.to(self.device, self.dtype)
                outputs = outputs.to(self.device, self.dtype)

                ## get tfunction and get loss
                pred_outputs = self.model.forward(inputs)
                loss = self.criterion(pred_outputs, outputs)

                ## log loss values
                loss_meter.update(loss.item())

                ## update params
                if is_train:
                    ## set grads to zero
                    self.optimizer.step()

                    ## get gradients and do optimizer step
                    loss.backward()
                    self.optimizer.step()

                ## update progress bar
                epoch_str = ('[Epoch %d / %d]' % (current_epoch + 1, self.total_epochs)) if current_epoch > -1 else ''
                pr_bar.set_description('%s Loss: %.4f (%.4f)' % (epoch_str, loss.item(), loss_meter.avg))

        self.log.log_learning_curve({'loss_%s' % ('train' if is_train else 'test'): loss_meter.val_list},
                                    plot=True)

        return loss_meter.avg
