import os
from os.path import join as join_path
from os.path import isfile
import _pickle as pickle
import json
#import shutil


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_args(folder, file_name='args.json'):
    file_path = join_path(folder, file_name)
    with open(file_path, 'r') as f:
        return json.load(f)


def save_args(args, folder, file_name='args.json', exclude_args=tuple()):
    save_args = {k:v for k,v in vars(args).items() if k not in exclude_args}
    file_path = join_path(folder, file_name)
    ensure_dir(file_path)
    with open(file_path, 'w') as f:
        return json.dump(save_args, f)


def save_summary(obj, path, rewrite=False):
    mode = 'bw' if rewrite else 'ba'
    with open(path, mode) as file:
        pickle.dump(obj, file, protocol=4)


def load_summary(path):
    data = []
    with open(path, 'br') as file:
        while True:
            try:
                data.extend(pickle.load(file))
            except EOFError:
                break
    return data


def clip_local_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norms of an iterable of parameters.

    The norms is computed over each parameter gradients separetly.
    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.
    for p in parameters:
        if norm_type == float('inf'):
            local_norm = p.grad.data.abs().max()
            total_norm = max(total_norm, local_norm)
        else:
            local_norm = p.grad.data.norm(norm_type)
            total_norm += local_norm ** norm_type

        clip_coef = max_norm / (local_norm + 1e-6)
        if clip_coef < 1.:
            p.grad.data.mul_(clip_coef)

    if norm_type != float('inf'):
        total_norm = total_norm ** (1./ norm_type)

    return total_norm


def global_norm(tensors, norm_type=2):
    """
    Returns the global norm of given tensors.
    The global norm is computed as:
        sum(p_norm**norm_type for p in parameters)**(1./norm_type)
    If norm_type equals to 'inf', then infinity norm is computeed:
        max(p.max() for p in parameters)
    """
    norm_type = float(norm_type)
    if norm_type == float('inf'):
      global_norm = max(t.max() for t in tensors)
    else:
      global_norm = sum(t.norm(norm_type)**norm_type for t in tensors)
      global_norm = global_norm**(1./norm_type)

    return global_norm


def global_grad_norm(parameters, norm_type=2):
    grads = [p.grad.data for p in parameters if p.grad is not None]
    return global_norm(grads, norm_type)


class BinaryClassificationStats(object):
    """
    BinaryClassificationStats doesn't update its data using running average.
    This class is supposed to be used in the evaluation process where the data comes
    in small batches, but the model doesn't perform training between those batches.
    """
    def __init__(self):
        self.true_pos = 0
        self.false_pos = 0
        self.true_neg = 0
        self.false_neg = 0

        self.targets_pos = 0
        self.targets_neg = 0
        self.preds_pos = 0
        self.preds_neg = 0

    def add_batch(self, preds, targets):
        self.true_pos += (preds * targets).sum()
        self.false_pos += (preds * (1 - targets)).sum()
        self.true_neg += ((1 - preds) * (1 - targets)).sum()
        self.false_neg += ((1 - preds) * targets).sum()

        self.preds_pos += preds.sum()
        self.preds_neg += (1 - preds).sum()
        self.targets_pos += targets.sum()
        self.targets_neg += (1 - targets).sum()

    @property
    def accuracy(self):
        correct = self.true_pos + self.true_neg
        all = correct + self.false_pos + self.false_neg
        if all > 0:
            return (correct / all)*100.
        return float('nan')

    @property
    def precision(self):
        all_pos_preds = self.true_pos + self.false_pos
        if all_pos_preds > 0:
            return (self.true_pos / all_pos_preds)*100.
        return float('nan')

    @property
    def recall(self):
        all_pos_targets = self.true_pos + self.false_neg
        if all_pos_targets > 0:
            return (self.true_pos / all_pos_targets)*100.
        return float('nan')

    @property
    def predictions_ratio(self):
        all_preds = self.preds_pos + self.preds_neg
        if all_preds > 0:
            return (self.preds_pos / all_preds)*100.
        return float('nan')

    @property
    def targets_ratio(self):
        all_targets = self.targets_pos + self.targets_neg
        if all_targets > 0:
            return  (self.targets_pos / all_targets)*100.
        return float('nan')

    @property
    def size(self):
        return self.preds_pos + self.preds_neg

    def pretty_print(self):
        print('BinaryClassificationStats:')
        print('Number of samples:', self.size)
        print('Accuracy: {0:.2f}%'.format(self.accuracy))
        print('Precision: {0:.2f}%'.format(self.precision))
        print('Recall: {0:.2f}%'.format(self.recall))
        print('targets_ratio: {0:.2f}%'.format(self.targets_ratio))
        print('predictions_ratio: {0:.2f}%'.format(self.predictions_ratio))


def red(line):
    return "\x1b[31;1m{0}\x1b[0m".format(line)

def yellow(line):
    return "\x1b[33;1m{0}\x1b[0m".format(line)