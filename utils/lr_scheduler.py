from torch.optim import Optimizer
import numbers

class LRScheduler(object):
    def __init__(self, optimizer, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
              '{0} is not an Optimizer'.format(type(optimizer))
            )
        self.optimizer = optimizer
        self.step = last_step

    def get_lr(self):
        raise NotImplementedError

    def adjust_learning_rate(self, step=None):
        if step is None:
            step = self.step + 1
        self.step = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    @staticmethod
    def _broadcast_for_param_groups(param, optimizer):
        if isinstance(param, numbers.Number):
            return [param for g in optimizer.param_groups]
        if len(param) == len(optimizer.param_groups):
            return param
        raise ValueError(
          'param should be a number of or' + \
          'a list of the same length as optimizer.param_groups'
        )


class LinearAnnealingLR(LRScheduler):
    def __init__(self, optimizer, annealing_steps, end_lr=0., last_step=-1):
        super(LinearAnnealingLR, self).__init__(optimizer, last_step)
        if last_step == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.init_lrs = [ g['initial_lr'] for g in optimizer.param_groups ]
        self.annealing_steps = annealing_steps
        self.lr_deltas = [(lr - end_lr)/annealing_steps for lr in self.init_lrs]
        self.end_lrs = self._broadcast_for_param_groups(end_lr, optimizer)

    def get_lr(self):
      if self.step >= self.annealing_steps:
        return self.end_lrs
      else:
        return [lr - self.step*d for lr, d in zip(self.init_lrs, self.lr_deltas)]
