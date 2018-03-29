import argparse
import copy
import logging

import numpy as np
import torch
from torch.autograd import Variable

import train_multi_task as tr
import utils
from paac import MultiTaskPAAC
from utils.lr_scheduler import LinearAnnealingLR


class TerminationModelRetrainer(MultiTaskPAAC):
    eval_every = 10240

    def __init__(self, network_creator, env_creator, args):
        self.args = copy.copy(vars(args))
        self.checkpoint_dir = utils.join_path(
            self.args['debugging_folder'], self.CHECKPOINT_SUBDIR
        )
        utils.ensure_dir(self.checkpoint_dir)
        self.global_step = self.last_saving_step = 0
        self.network = network_creator()
        self.optimizer = torch.optim.RMSprop(
            self.network.terminal_prediction_params(),
            lr=self.args['initial_lr'],
            alpha=self.args['alpha'],
            eps=self.args['e'],
            weight_decay=self.args['reg_coef']
        )

        self.lr_scheduler = LinearAnnealingLR(
            self.optimizer,
            self.args['lr_annealing_steps'],
            end_lr=self.args['end_lr']
        )

        self.use_cuda = self.args['device'] == 'gpu'
        self.use_lstm = self.args['arch'] == 'lstm'
        self._modeltypes = torch.cuda if self.use_cuda else torch
        self._looptypes = self._modeltypes # model

        self.action_codes = np.eye(self.args['num_actions'])
        self.gamma = self.args['gamma']  # future rewards discount factor
        self.emulators = np.asarray(
            [env_creator.create_environment(i) for i in range(self.args['num_envs'])]
        )

        if self.args['clip_norm_type'] == 'global':
            self.clip_gradients = torch.nn.utils.clip_grad_norm
        elif self.args['clip_norm_type'] == 'local':
            self.clip_gradients = utils.clip_local_grad_norm
        elif self.args['clip_norm_type'] == 'ignore':
            self.clip_gradients = lambda params, _: utils.global_grad_norm(params)
        else:
            raise ValueError('Norm type({}) is not recoginized'.format(self.args['clip_norm_type']))

        self.EVAL_EVERY = args.eval_every
        self.last_eval = self.global_step - self.EVAL_EVERY
        self._obs_shape = env_creator.obs_shape
        self._preprocess_states = env_creator.preprocess_states
        self._term_model_coef = args.termination_model_coef
        self._term_model_loss = torch.nn.NLLLoss(weight=torch.FloatTensor([0.66, 1.33]))
        if self.use_cuda:
            self._term_model_loss = self._term_model_loss.cuda()


    def compute_loss(self, delta_v, selected_log_probs, entropies):
        """No Actor-Critic Loss"""
        loss = torch.zeros(1,).type(self._modeltypes.FloatTensor)
        return Variable(loss), Variable(loss.clone()), Variable(loss.clone())


    def compute_termination_model_loss(self, log_terminals, tasks):
        loss = super(TerminationModelRetrainer, self).compute_termination_model_loss(
            log_terminals, tasks
        )
        if (self.global_step - self.last_eval) >= self.EVAL_EVERY:
            self.last_eval = self.global_step
            targets = (tasks[:-1] != tasks[1:]).astype(int)
            targets = torch.from_numpy(targets).view(-1).type(self._modeltypes.LongTensor)
            preds = torch.cat(log_terminals, 0).data
            preds = torch.max(preds, dim=1)[1]

            target_ratio = targets.sum() / targets.size(0)
            pred_ratio = preds.sum() / preds.size(0)

            true_pos = (preds * targets).sum()
            false_pos = (preds * (1-targets)).sum()
            true_neg = ((1-preds) * (1-targets)).sum()
            false_neg = ((1-preds) * targets).sum()
            accuracy = (true_pos + true_neg)/(true_pos+true_neg+false_neg+false_pos)
            # точность: доля парвильных окончаний задачи из всех попыток её закончить
            precision = true_pos/(true_pos + false_pos + 1e-8)
            # полнотя: доля случаев когда агент смог распознать правильное окончание задачи
            recall =true_pos/(true_pos + false_neg + 1e-8)
            logging.debug(red('STEP: {0}'.format(self.global_step)))
            logging.debug(red("Predictor accuracy = {0:.2f}".format(accuracy*100)))
            logging.debug(red(
                'Predictor precision = {0:.2f}, recall = {1:.2f}'.format(precision,recall)
            ))
            logging.debug(red(
                'target done ratio = {0}, pred done ratio = {1}'.format(target_ratio, pred_ratio)
            ))
        return loss

def main(args):
    checkpoint_dir = utils.join_path(args.path, MultiTaskPAAC.CHECKPOINT_SUBDIR)
    checkpoint = MultiTaskPAAC._load_latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None, "Can't find a pretrained agent!"

    network_creator, env_creator = tr.get_network_and_environment_creator(args)

    def pretrained_net_creator():
        network = network_creator()
        logging.info('Loading a pretrained agent!')
        logging.info('The agent was trained for {} steps'.format(checkpoint['last_step']))
        network.load_state_dict(checkpoint['network_state_dict'])
        return network

    logging.info(tr.args_to_str(args))
    logging.info('Initializing PAAC...')

    learner = TerminationModelRetrainer(pretrained_net_creator, env_creator, args)

    tr.concurrent_emulator_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, dest='path',
                        help='Path for a pretrained multi-task agent!')
    parser.add_argument('-d', '--device', type=str, default='gpu', choices=['gpu', 'cpu'],
                        help='Computational device to train with')
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-lr', '--initia_lr', default=0.0075, type=float, dest="initial_lr",
                        help="Initial value for the learning rate. Default = 0.0075")
    parser.add_argument('--e', default=0.1, type=float,  dest="e",
                        help="Epsilon for the Rmsprop and Adam optimizers",)
    parser.add_argument('-elr', '--end_lr', default=0., type=float, dest='end_lr',
                       help='During training initial_lr will be linearly annealed towards end_value')
    parser.add_argument('--max_global_steps', type=int, default=80000000)
    parser.add_argument('-n', '--num_envs', default=32, type=int,
                        help="The amount of emulators per agent. Default is 32.")
    parser.add_argument('-w', '--workers', default=8, type=int, dest="num_workers",
                        help="The amount of emulator workers per agent. Default is 8.")
    parser.add_argument('-r', '--reg_coef', default=0., type=float, dest='reg_coef',
                        help='Scalar giving L2 regularization strength')
    parser.add_argument('--eval_every', default=10240, type=int, dest='eval_every',
                        help='Model evaluation frequency.')

    return parser


def update_args(args, agent_args):
    for k, v in agent_args.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.loss_scaling = 0. # we don't want to retrain our agent so a3c loss should b equal to zero
    args.termination_model_coef = 1. # we will train the termination prediction loss
    args.lr_annealing_steps = args.max_global_steps
    args.clip_norm_type = 'ignore'


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    agent_args = utils.load_args(folder=args.path, file_name=tr.ARGS_FILE)
    update_args(args, agent_args)

    logging.info('Start training')
    main(args)
    logging.info('Done.')
