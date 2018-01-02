from paac import *
from collections import namedtuple
from utils import red, yellow

class MovingAverage(object):
    def __init__(self, update_coef, averages_names):
        self.avr_dict = {n:0. for n in averages_names}
        self.update_coef = update_coef

    def update(self, **new_data):
        old_coef = 1. - self.update_coef
        for k, val in new_data.items():
            self.avr_dict[k] = self.update_coef*val + old_coef*self.avr_dict[k]

    def __getattr__(self, item):
        return self.avr_dict[item]

    def __str__(self):
        l = ['MovingAverage:']
        for k, v in self.avr_dict.items():
            l.append('{}={}'.format(k,v))
        return ' '.join(l)

TrainingStats = namedtuple("TrainingStats",
                           ['mean_r','max_r','min_r','std_r',
                            'mean_steps','term_acc','term_rec',
                            'term_prec','t_ratio', 'p_ratio'])

class MultiTaskPAAC(PAACLearner):
    EVAL_EVERY = 10240

    def __init__(self, network_creator, env_creator, args):
        super(MultiTaskPAAC, self).__init__(network_creator, env_creator, args)
        self._preprocess_states = lambda states: env_creator.preprocess_states(states, env_creator.obs_shape)
        self.env_creator = env_creator

        self._term_model_coef = args.termination_model_coef
        logging.debug('Termination loss class weights = {0}'.format(args.term_weights))
        self._term_model_loss = nn.NLLLoss(weight=torch.FloatTensor(args.term_weights))
        if self.use_cuda:
            self._term_model_loss = self._term_model_loss.cuda()
        if hasattr(args, 'eval_every'):
            self.EVAL_EVERY = args.eval_every
        self.eval_func = None

    def train(self):
        """
         Main actor learner loop for parallerl advantage actor critic learning.
         """
        logging.info('Starting training at step %d' % self.global_step)
        logging.debug('Tensor types. Model: {}, Loop: {}'.format(
            self._modeltypes.FloatTensor, self._looptypes.FloatTensor
        ))

        counter = 0
        global_step_start = self.global_step
        ma_loss = MovingAverage(0.01, ['total', 'actor', 'critic', 'term_model'])
        total_rewards, training_stats = [], []

        if self.eval_func is not None:
            stats = self.evaluate(verbose=True)
            training_stats.append((self.global_step, stats))

        # num_actions = self.args['num_actions']
        num_emulators = self.args['emulator_counts']
        max_local_steps = self.args['max_local_steps']
        max_global_steps = self.args['max_global_steps']
        rollout_steps = num_emulators * max_local_steps
        clip_norm = self.args['clip_norm']

        self.runners = self._create_runners()
        self.runners.start()

        shared_vars = self.runners.get_shared_variables()
        shared_s, shared_r, shared_done, shared_a = shared_vars
        # any summaries here?
        # actions_sum = np.zeros((num_emulators, num_actions))
        emulator_steps = np.zeros(num_emulators, dtype=int)
        total_episode_rewards = np.zeros(num_emulators)
        not_done_masks = torch.zeros(max_local_steps, num_emulators).type(self._looptypes.FloatTensor)
        tasks = np.zeros((max_local_steps+1, num_emulators))

        if self.use_lstm:
            hx_init, cx_init = self.network.get_initial_state(num_emulators)
            hx, cx = hx_init.detach(), cx_init.detach()  # Do I really need to detach here?
        #extra_inputs = {} # additional inputs needed to different types of networks
        start_time = time.time()
        while self.global_step < max_global_steps:
            loop_start_time = time.time()
            values, log_probs, rewards, entropies, log_terminals = [], [], [], [], []

            if self.use_lstm:
                hx, cx = hx.detach(), cx.detach()  # Do I really need to detach here?

            # print('outer loop #{} global_step#{}'.format(counter, self.global_step))
            for t in range(max_local_steps):
                s_t, task_t = self._preprocess_states(shared_s)
                if self.use_lstm:
                    a_t, v_t, log_probs_t, entropy_t, log_term_t, (hx, cx) = self.choose_action(s_t, task_t, rnn_inputs=(hx,cx))
                else:
                    a_t, v_t, log_probs_t, entropy_t, log_term_t = self.choose_action(s_t, task_t)

                shared_a[:] = a_t[:]
                self.runners.update_environments()
                self.runners.wait_updated()
                # actions_sum += a_t
                tasks[t] = task_t
                log_terminals.append(log_term_t)
                rewards.append(np.clip(shared_r, -1., 1.))
                entropies.append(entropy_t.type(self._looptypes.FloatTensor))
                log_probs.append(log_probs_t.type(self._looptypes.FloatTensor))
                values.append(v_t.type(self._looptypes.FloatTensor))
                is_done = torch.from_numpy(shared_done).type(self._looptypes.FloatTensor)
                not_done_masks[t] = 1.0 - is_done

                done_mask = shared_done.astype(bool)
                total_episode_rewards += shared_r
                emulator_steps += 1
                self.global_step += num_emulators
                total_rewards.extend(total_episode_rewards[done_mask])
                total_episode_rewards[done_mask] = 0.
                emulator_steps[done_mask] = 0
                if self.use_lstm and any(done_mask):  # we need to clear all lstm states corresponding to the terminated emulators
                    done_idx = is_done.nonzero().view(-1)
                    hx, cx = hx.clone(), cx.clone()  # hx_t, cx_t are used for backward op, so we can't modify them in-place
                    hx[done_idx, :] = hx_init[done_idx, :].detach()
                    cx[done_idx, :] = cx_init[done_idx, :].detach()
                    # print('  inner_loop #{0} global_step#{1}'.format(t, self.global_step))

            rnn_inputs = (hx, cx) if self.use_lstm else None
            next_s, next_task = self._preprocess_states(shared_s)
            tasks[max_local_steps] = next_task
            next_v = self.predict_values(next_s, next_task, rnn_inputs=rnn_inputs)

            R = next_v.detach().view(-1).type(self._looptypes.FloatTensor)

            delta_v = []
            for t in reversed(range(max_local_steps)):
                r_t = Variable(torch.from_numpy(rewards[t])).type(self._looptypes.FloatTensor)
                not_done_t = Variable(not_done_masks[t])
                R = r_t + self.gamma * R * not_done_t
                delta_v_t = R - values[t].view(-1)
                delta_v.append(delta_v_t)

            loss, actor_loss, critic_loss = self.compute_loss(
                torch.cat(delta_v, 0),
                torch.cat(log_probs, 0).view(-1),
                torch.cat(entropies, 0).view(-1)
            )

            term_loss = self.compute_termination_model_loss(log_terminals, tasks)
            if self._term_model_coef > 0. and self.global_step >= self.args['warmup']:
                loss += self._term_model_coef * term_loss

            self.lr_scheduler.adjust_learning_rate(self.global_step)
            self.network.zero_grad()
            loss.backward()
            global_norm = self.clip_gradients(self.network.parameters(), clip_norm)
            self.optimizer.step()

            ma_loss.update(total=loss.data[0], actor=actor_loss.data[0],
                           critic=critic_loss.data[0], term_model=term_loss.data[0])
            counter += 1
            if counter % (10240 // rollout_steps) == 0:
                curr_time = time.time()
                self._training_info(
                    total_rewards=total_rewards,
                    average_speed=(self.global_step-global_step_start) / (curr_time-start_time),
                    loop_speed=rollout_steps / (curr_time-loop_start_time),
                    moving_averages=ma_loss, grad_norms=global_norm
                )
            if counter % (self.EVAL_EVERY // rollout_steps) == 0:
                if (self.eval_func is not None):
                    stats = self.evaluate(verbose=True)
                    training_stats.append((self.global_step, stats))

            if self.global_step - self.last_saving_step >= self.CHECKPOINT_INTERVAL:
                self._save_progress(
                    self.network, self.optimizer, self.checkpoint_dir,
                    summaries=training_stats, is_best=False,
                )
                training_stats = []
                self.last_saving_step = self.global_step

        self.cleanup()
        logging.debug('Training ended at step %d' % self.global_step)

    def choose_action(self, *inputs, **kwargs):
        if self.use_lstm:
            values, a_logits, done_logits, lstm_state = self.network(*inputs, **kwargs)
        else:
            values, a_logits, done_logits = self.network(*inputs)

        log_done = F.log_softmax(done_logits, dim=1)
        probs = F.softmax(a_logits, dim=1)
        log_probs = F.log_softmax(a_logits, dim=1)
        entropy = torch.neg((log_probs * probs)).sum(1)
        acts = probs.multinomial().detach()
        selected_log_probs = log_probs.gather(1, acts)

        check_log_zero(log_probs.data)
        acts_one_hot = self.action_codes[acts.data.cpu().view(-1).numpy(), :]
        if self.use_lstm:
            return acts_one_hot, values, selected_log_probs, entropy, log_done, lstm_state
        else:
            return acts_one_hot, values, selected_log_probs, entropy, log_done,

    def predict_values(self, *inputs, **extra_inputs):
        if self.use_lstm:
            values = self.network(*inputs, **extra_inputs)[0]
        else:
            values = self.network(*inputs)[0]
        return values

    def set_eval_function(self, eval_func, args, kwargs):
        self.eval_func = eval_func
        self.eval_args = args
        self.eval_kwargs = kwargs

    def evaluate(self, verbose=True):
        prev_mode = self.network.training
        self.network.eval() # no need to save additional information needed for training
        num_steps, rewards, term_stats = self.eval_func(*self.eval_args, **self.eval_kwargs)
        self.network.train(prev_mode)

        mean_steps = np.mean(num_steps)
        min_r, max_r = np.min(rewards), np.max(rewards)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        acc = term_stats.accuracy
        rec = term_stats.recall
        prec = term_stats.precision
        targets_ratio = term_stats.targets_ratio
        preds_ratio = term_stats.predictions_ratio

        stats = TrainingStats(
            mean_r=mean_r, min_r=min_r, max_r=max_r, std_r=std_r,
            term_acc=acc, term_prec=prec, term_rec=rec,
            mean_steps=mean_steps, t_ratio=targets_ratio, p_ratio=preds_ratio
        )

        if verbose:
            lines = [
                'Perfromed {0} tests:'.format(len(num_steps)),
                'Mean number of steps: {0:.3f}'.format(mean_steps),
                'Mean R: {0:.2f} | Std of R: {1:.3f}'.format(mean_r, std_r),
                'Termination Predictor:',
                'Acc: {:.2f}% | Precision: {:.2f} | Recall: {:.2f}'.format(acc, prec, rec),
                'Class 1 ratio. Targets: {0:.2f}% Preds: {1:.2f}%'.format(targets_ratio, preds_ratio)]
            logging.info(red('\n'.join(lines)))

        return stats

    def compute_termination_model_loss(self, log_terminals, tasks):
        tasks_done = (tasks[:-1] != tasks[1:]).astype(int)
        tasks_done = torch.from_numpy(tasks_done).type(self._modeltypes.LongTensor)
        tasks_done = Variable(tasks_done.view(-1))
        log_terminals = torch.cat(log_terminals, 0).type(self._modeltypes.FloatTensor)
        term_loss = self._term_model_loss(log_terminals, tasks_done)
        return term_loss

    def _training_info(self, total_rewards, average_speed, loop_speed, moving_averages, grad_norms):
        last_ten = np.mean(total_rewards[-10:]) if len(total_rewards) else 0.
        logger_msg = "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"

        lines = ['',]
        lines.append(logger_msg.format(self.global_step, loop_speed, average_speed, last_ten))
        lines.append(str(moving_averages))
        lines.append('grad_norm: {}'.format(grad_norms))
        logging.info(yellow('\n'.join(lines)))
