from .multi_task_paac import MultiTaskActorCritic
import torch.optim as optim
from utils.lr_scheduler import LinearAnnealingLR
import logging
from utils import join_path, ensure_dir




class LifelongMTActorCritic(MultiTaskActorCritic):

    def __init__(self, *args, **kwargs):
        retrain_modules = kwargs.pop(
            'retrain_modules',
            set('task_lstm.embedding',
                'fc_terminal',
                'fc_value')
        )
        super(LifelongMTActorCritic, self).__init__(*args, **kwargs)
        train_params, freeze_params = self._split_parameters(self, retrain_modules)
        for name, p in freeze_params:
            print('freeze parameter:', name)
            p.requires_grad = False
        self.global_step = 0

        self.optimizer.RMSprop(
            [p for n, p in train_params],
            lr=args.initial_lr,
            eps=args.e,
        )

        self.lr_scheduler = LinearAnnealingLR(
            self.optimizer,
            args.lr_annealing_steps,
        )


    def _split_parameters(self, retrain_modules):
        """
        Return a 2-tuple containing a list of parameters
        for training and a list of parameters that will be frozen
        """
        # We assume that parameter names have the following format:
        #higher_module.lower_module.parameter_name
        get_module_name = lambda p: name.rpartition('.')[0]

        train, freeze = [], []
        for name, param in self.network.named_parameters():
            if get_module_name(name) in retrain_modules:
                train.append((name, param))
            else:
                freeze.append((name, param))

        return train, freeze