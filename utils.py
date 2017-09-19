import os
from os.path import join as join_path
from os.path import isfile
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


def save_args(args, folder, file_name='args.json'):
    args = vars(args)
    file_path = join_path(folder, file_name)
    ensure_dir(file_path)
    with open(file_path, 'w') as f:
        return json.dump(args, f)


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
