from .multi_task_paac import MultiTaskActorCritic
import logging, time
import torch as th



class CurriculumMTActorCritic(MultiTaskActorCritic):

    def __init__(self,
                 network,
                 batch_env,
                 curriculum_manager,
                 args):
        super(CurriculumMTActorCritic, self).__init__(network, batch_env, args)
        self.curriculum_manager = curriculum_manager

    def set_curriculum_manager(self, manager, update_every):
        self.curriculum_update_freq = update_every
        self.curriculum_manager = manager

    def train(self, ):
        """
        Main actor learner loop for parallerl advantage actor critic learning.
        """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('Device: {}'.format(self.device))

        num_updates = 0
        global_step_start = self.global_step

        num_emulators = self.batch_env.num_emulators
        training_stats = []
        steps_per_update = num_emulators * self.rollout_steps
        curr_mean_r = best_mean_r = float('-inf')

        if self.evaluate:
            stats = self.evaluate(self.network)
            training_stats.append((self.global_step, stats))
            curr_mean_r = best_mean_r = stats.mean_r

        state, info = self.batch_env.reset_all()
        #stores 0.0 in i-th element if the episode in i-th emulator has just started, otherwise stores 1.0
        mask = th.zeros(self.batch_env.num_emulators).to(self.device)
        #feedforward networks also use rnn_state, it's just empty!
        rnn_state = self.network.init_rnn_state(num_emulators)

        start_time = time.time()
        while self.global_step < self.total_steps:

            loop_start_time = time.time()
            rollout_data, finals = self.rollout(state, info, mask, rnn_state)
            #final states of environments and network at the end of rollout
            state, info, mask, rnn_state = finals

            update_stats = self.update_weights(rollout_data)
            self.average_loss.update(**update_stats)

            self.global_step += steps_per_update
            num_updates += 1

            if num_updates % (self.print_every // steps_per_update) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=self.reward_history,
                    average_speed=(self.global_step - global_step_start) / (curr_time - start_time),
                    loop_speed=steps_per_update / (curr_time - loop_start_time),
                    update_stats=self.average_loss)

            if num_updates % (self.eval_every // steps_per_update) == 0:
                if self.evaluate:
                    stats = self.evaluate(self.network)
                    training_stats.append((self.global_step, stats))
                    curr_mean_r = stats.mean_r

            if self.global_step - self.last_saving_step >= self.save_every:
                is_best = False
                if curr_mean_r > best_mean_r:
                    best_mean_r = curr_mean_r
                    is_best = True
                self._save_progress(self.checkpoint_dir, summaries=training_stats, is_best=is_best)
                training_stats = []
                self.last_saving_step = self.global_step

        self._save_progress(self.checkpoint_dir, is_best=False)
        logging.info('Training ended at step %d' % self.global_step)
