import os
import shutil
import plotter
import json
import inspect
import torch
import datetime


def _merge(old_dict, new_dict):
    """
    merges two dicts, mantaining the previous values in their keys and generating new keys
    if they did not exist before
    """
    dict3 = old_dict.copy()
    for k, v in new_dict.items():
        if k in dict3:
            dict3[k].append(v)
        else:
            dict3[k] = [v]
    return dict3


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/single.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val_list = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 9999999
        self.max = 0
        self.squared_sum = 0

    def update(self, val, n=1):
        self.val_list.append(val)
        self.val = val
        self.sum += val * n
        self.squared_sum += val ** 2 * n
        self.count += n
        self.variance = self.squared_sum / self.count
        self.avg = self.sum / self.count
        self.min = val if val < self.min else self.min
        self.max = val if val > self.max else self.max


class Logger:
    def __init__(self, logpath):
        now = datetime.datetime.now()

        logpath_split = logpath.split('/')
        logpath_split[-1] = ('[%02d-%02d-%04d_%02d:%02d]' %
                             (now.day, now.month, now.year, now.hour, now.minute)) + logpath_split[-1]
        self.logpath = '/'.join(logpath_split)

        self.logpath_models = os.path.join(self.logpath, 'models')
        self.logpath_images = os.path.join(self.logpath, 'images')
        self.logpath_training = os.path.join(self.logpath, 'training_data.json')
        self.logpath_transforms = os.path.join(self.logpath, 'data_transforms.log')

        ## create necessary dirs and files
        os.makedirs(self.logpath_models, exist_ok=True)
        os.makedirs(self.logpath_images, exist_ok=True)
        with open(self.logpath_training, 'w') as f:
            json.dump({}, f)

    def log_transforms(self, train_loader, test_loader):
        def _store_transform(trf_str, trf_list):
            for trf in trf_list:
                if trf is None:
                    trf_str += '\t\tNone\n'
                else:
                    trf_str += inspect.getsource(trf)
            return trf_str

        trf_input_train = train_loader.dataset.trf_input
        trf_output_train = train_loader.dataset.trf_output
        trf_input_test = test_loader.dataset.trf_input
        trf_output_test = test_loader.dataset.trf_output

        transforms_log = '=============================\nTRAIN\n============================='
        transforms_log = _store_transform(transforms_log + '\nTRANSFORMS_INPUT\n', trf_input_train)
        transforms_log = _store_transform(transforms_log + 'TRANSFORMS_OUTPUT\n', trf_output_train)
        transforms_log = transforms_log + '\n=============================\nTEST\n============================='
        transforms_log = _store_transform(transforms_log + '\nTRANSFORMS_INPUT\n', trf_input_test)
        transforms_log = _store_transform(transforms_log + 'TRANSFORMS_OUTPUT\n', trf_output_test)
        with open(self.logpath_transforms, 'w') as f:
            f.write(transforms_log)

    def log_learning_curve(self, new_data, plot=False):
        """
        stores the values given in a dict (new_data) in a json file for each iteration
        if plot is true, it will plot those values in a curve.
        new_data is usually a dict that contains the metrics of the model in each iteration
        """
        with open(self.logpath_training, 'r') as f:
            data = json.load(f)

        data = _merge(data, new_data)

        with open(self.logpath_training, 'w') as f:
            json.dump(data, f, sort_keys=True)

        if plot:
            plotter.plot_learning_curve(data)

    def save_model(self, is_best, state, epoch):
        """
        Saves the current model (state) in a given file. If it is the best
        model (so far) it creates a copy of it with name 'model_best'
        """
        path = os.path.join(self.logpath_models, 'model-%d.pth.tar' % epoch)
        torch.save(state, path)
        if is_best:
            shutil.copyfile(path, path + 'model_best.pth.tar')

    def save_plot(self, ):
        """
        Save a plot in the images folder
        :return:
        """
        pass
