import argparse
import random
import numpy as np
import torch
import torch.utils.data as torch_data
from data_reader import YourDataset
import runner
import models
from tqdm import tqdm
import losses
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seeds(seed):
    # set the manual seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_class(x):
    return x.__class__.__name__


def get_params():
    parser = argparse.ArgumentParser(description='Paint-Appearance',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Config params
    parser.add_argument('--dir', default='/media/data_3/paint_appearance/fixed_WB_data_16', type=str,
                        help='dataset folder')
    parser.add_argument('--workers', default=4, type=int, help='data loading workers')

    parser.add_argument('--plot_iter_train', default=50, type=int, help='number of iterations to plot a result during training')
    parser.add_argument('--plot_iter_test', default=30, type=int, help='number of iterations to plot a result during test')

    # Model params
    parser.add_argument('--total-epoch', default=50, type=int, help='number of epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number ')

    # Adabound params
    parser.add_argument('--batch-size', default=2, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--gamma', default=1e-3, type=float, help='convergence speed term of AdaBound')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay for optimizers')

    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--seed', default=1325, type=int, help='manual seed')

    return parser


if __name__ == "__main__":
    ## get args and set default arguments
    args = get_params().parse_args()

    ## set seeds for random generation
    set_seeds(args.seed)

    ## initialize the dataset and data-loaders
    tqdm.write('loading datasets...')
    trf_input = lambda x: torch.from_numpy(x)
    data_train = YourDataset(args.dir, trf_input, is_train=True)
    data_test = YourDataset(args.dir, trf_input, is_train=False)

    loader_args = {'num_workers': args.workers, 'pin_memory': True, 'batch_size': args.batch_size}
    train_loader = torch_data.DataLoader(dataset=data_train, shuffle=True, drop_last=True, **loader_args)
    test_loader = torch_data.DataLoader(dataset=data_test, **loader_args)

    ## create the model
    tqdm.write('creating model...')
    model = models.YourModel()

    ## create the criterion
    tqdm.write('creating criterions...')
    criterion = losses.SquaredDifferences(reduction='batch-mean')

    ## set the optimizers
    tqdm.write('creating optimizers...')
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)

    ## create logger
    logpath = 'checkpoints/mytraining-%s-%s-%s' % \
              (get_class(model), get_class(criterion), get_class(optimizer))
    tqdm.write('checkpoints will be saved at: %s' % logpath)

    ## define the learning abstraction
    learning = runner.RunnerExample(
        model=model,  # models
        criterion=criterion,  # criterion
        optimizer=optimizer,  # optimizer for the parameters
        train_loader=train_loader, test_loader=test_loader,  # data
        total_epochs=args.total_epoch, start_epoch=args.start_epoch,  # training epochs
        logpath=logpath,  # logger info
        resume_path=args.resume
    )

    if not args.evaluate:
        ## run the training loop
        tqdm.write('start training...\n')
        learning.fit(plot_iter_train=args.plot_iter_train, plot_iter_test=args.plot_iter_test)
    else:
        ## evaluate model
        learning.do_iteration(is_train=False)
