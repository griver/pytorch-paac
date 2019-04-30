import logging
from multiprocessing import Queue, JoinableQueue, Pipe
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double, c_int32, c_bool
import numpy as np
import os, signal
from typing import Tuple, List, Any

NUMPY_TO_C_DTYPE = {
    np.float32: c_float,
    float: c_float,
    np.float64: c_double,
    np.uint8: c_uint,
    np.int32: c_int32,
    np.int64: c_int32,
    int: c_int32,
    bool: c_bool,
}


class VariableTypeException(Exception): pass


class BatchEmulatorError(Exception): pass


def get_shared(array):
    """
    Returns a RawArray backed numpy array that can be shared between processes.
    :param array: the array to be shared
    :return: the RawArray backed numpy array
    """
    dtype = NUMPY_TO_C_DTYPE[array.dtype.type]

    shape = array.shape
    shared = RawArray(dtype, array.reshape(-1))
    return np.frombuffer(shared, dtype).reshape(shape)


def shape_and_dtype(var):
    """returns shape of the variable and type of its elements"""
    if np.isscalar(var):
        return (), type(var)
    elif isinstance(var, (np.ndarray, tuple, list)):
        var = np.asarray(var)
        return var.shape, var.dtype.type
    else:
        raise VariableTypeException(
            "shape_and_dtype works only with scalars and numpy arrays."
            " {} was given.".format(type(var)))


class BaseBatchEnv(object):

    def __init__(self, env_creator, num_emulators):
        self.num_emulators = num_emulators
        self.num_actions = env_creator.num_actions
        self.obs_shape = env_creator.obs_shape

    def _create_variables(self, env_creator, extra_vars):
        """
        Сreates a numpy array for each variable required for interaction
        between a learning algorithm and the emulators
        :param extra_vars: a list of extra variables to collect from emulators aside
                          from state, reward and is_done signals.
        :return: A dict with input variables(action),
                 a dict with output variables(state, reward, is_done),
                 a dict with the extra_vars.
                 All dicts' values are numpy arrays.
        """
        num_em = self.num_emulators
        example_em = env_creator.create_environment(-1, visualize=False, verbose=0)
        obs, info = example_em.reset()
        obs_dtype = obs.dtype

        input_vars = {
            'action': np.zeros((num_em,),dtype=np.int32) #actions are assumed to be discrete so np.int32
        }
        output_vars = {
            'state':np.zeros((num_em,)+ self.obs_shape, dtype=obs_dtype),
            'reward':np.zeros(num_em, dtype=np.float32),
            'is_done':np.asarray([False] * num_em, dtype=np.float32),
        }

        if extra_vars == 'all':
            extra_vars = list(info.keys()) if info != None else ()

        extra_vars = {var:None for var in extra_vars}
        for var in extra_vars.keys():
            if var not in info:
                raise BatchEmulatorError(
                    "{} is required but the emulators don't provide it!".format(var))
            var_shape, dtype = shape_and_dtype(info[var])
            var_shape = (num_em,) + var_shape
            extra_vars[var] = np.zeros(var_shape, dtype=dtype)

        example_em.close()
        return input_vars, output_vars, extra_vars

    def next(self, action):
        raise NotImplementedError()

    def reset_all(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class SharedMemBatchEnv(BaseBatchEnv):
    """
    SharedMemBatchEnv creates <num_workers> worker processes.
    Each worker uses an approximate equal emulators' share and sequentially updates them.
    """
    def __init__(self, worker_cls, env_creator, num_workers,
                 num_emulators, extra_vars='all'):
        """
        :param worker_cls: A class for the worker processes
        :param env_creator: Creates new environments
        :param num_workers: A number of concurrently working processes.
        :param num_emulators: A number of game environments that will be played simultaneously
        :param extra_vars: A tuple of extra variables names
                           or 'all' if you want to get all variables from the info dicts returned by the emulators.
        """
        super(SharedMemBatchEnv, self).__init__(env_creator, num_emulators)
        self.num_workers = num_workers
        self._command = worker_cls.Command

        inputs, outputs, extra_outputs = self._create_variables(env_creator, extra_vars)
        for k, array in inputs.items(): # default inputs: action
            setattr(self, k, get_shared(array))
        for k, array in outputs.items(): # default outputs: state, reward, is_done
            setattr(self, k, get_shared(array))
        self.info = {k:get_shared(array) for k, array in extra_outputs.items()}

        self.worker_queues = [Queue() for _ in range(num_workers)]
        self.barrier = Queue()
        self.workers, self.emulators_slices = self._create_workers(env_creator, worker_cls=worker_cls)
        self.is_running = False
        self.is_closed = False

    def _create_workers(self, env_creator, worker_cls):
        """
        Creates self.num_workers worker processes.
        Each worker receives an approximate equal share of emulators and corresponding variables.
        All Workers run concurrently but each of them updates their emulators sequentially.
        :return: A list of created workers
        """
        if self.num_emulators < self.num_workers:
            raise BatchEmulatorError(
                "{}: Number of emulators must be bigger than number of workers".format(type(self).__name__)
            )

        workers = [None]*self.num_workers
        # emulators_slices[i] will store number of emulators being processed by the i-th worker:
        emulators_slices = [None]* self.num_workers
        #following segment creates workers and splits emulators between them as fairly as possible:
        min_local_ems = self.num_emulators // self.num_workers
        extra_ems = self.num_emulators % self.num_workers
        l = r = 0

        for wid in range(self.num_workers):
            num_ems = min_local_ems + int(wid < extra_ems)
            l, r = r, min(r+num_ems, self.num_emulators)

            emulators_slices[wid] = num_ems

            worker_vars = {
                'action': self.action[l:r], 'state': self.state[l:r],
                'is_done': self.is_done[l:r],'reward': self.reward[l:r],
            }
            worker_extra_vars = {k:v[l:r] for k, v in self.info.items()}
            create_ems = lambda l=l,r=r: [env_creator.create_environment(i) for i in range(l,r)]

            workers[wid] = worker_cls(
                wid, create_ems, self.worker_queues[wid],
                self.barrier, worker_vars, worker_extra_vars
            )

        return workers, emulators_slices

    def start_workers(self):
        """
        Starts worker processes.
        Despite start_workers resembles the reset method from emulator classes,
        it doesn't reset emulators to new episodes.
        Ideally, start_worker should be called only once before the training stage.
        """
        if self.is_closed:
            raise BatchEmulatorError('{} is already closed'.format(type(self)))
        if not self.is_running:
            self.is_running = True
            for r in self.workers:
               r.start()

    def stop_workers(self):
        """Just use close()"""
        if self.is_closed:
            raise BatchEmulatorError('{} is already closed'.format(type(self)))
        if self.is_running and not self.is_closed:
            for queue in self.worker_queues:
                queue.put((self._command.CLOSE, None))
            self.is_running = False

    def close(self):
        """Stops worker processes and joins them."""
        if not self.is_closed:
            self.stop_workers()
            for worker in self.workers:
                worker.join()
            self.is_closed = True

    def next(self, action):
        """
        Performs given actions on the corresponding emulators i.e. performs action[i] on emulator[i].
        :param action:  Array of actions. if action space is discrete one-hot encoding is used.
        :return: states, rewards, dones, infos
        """
        self.action[:] = action
        #send signals to workers to update their environments(emulators)
        for queue in self.worker_queues:
            queue.put((self._command.NEXT, None))
        #wait until all emulators are updated:
        for _ in self.workers:
            self.barrier.get()
        return self.state, self.reward, self.is_done, self.info

    def reset_all(self):
        #print('{} Send RESET:'.format(type(self).__name__))
        for queue in self.worker_queues:
            queue.put((self._command.RESET, None))
        for _ in self.workers:
            self.barrier.get()
        return self.state, self.info

    def call_method(self, method_name: str, method_args: List[Tuple[list, dict]]=None) -> List[Any]:
        """
        Calls an arbitrary method for all running emulators
        This call_method works slower than next and reset_all methods as it doesn't make use of arrays in shared memory.
        Use this if you need to interact with the emulated environments in some unusual manner.
        But try to call this method as rare as possible if you don't want to slower your learning fps.
        :arg method_name: name of the method you want to call
        :arg method_args: list of shape [(emulator1_args, emulator1_kwargs),(emulator2_args, emulator2_kwargs), ...]
        """
        cmd = self._command.CALL_METHOD

        for w_id, (l,r) in enumerate(self.emulators_slices):
            data = {'arg_list:':method_args[l:r], 'method_name':method_name}
            self.worker_queues[w_id].put((cmd, data))

        results = []
        for _ in self.workers:
            results.extend(self.barrier.get())
        return results


class ConcurrentBatchEnv(BaseBatchEnv):
    def __init__(
        self,
        worker_cls,
        env_creator,
        num_workers,
        num_emulators,
    ):
        self.num_workers = num_workers
        self.num_emulators = num_emulators
        self._command = worker_cls.Command

        self.conns, self.worker_conns = zip(*[Pipe() for _ in range(self.num_workers)])

        self.workers, self.emulators_slices = self._create_workers(env_creator, worker_cls=worker_cls)
        self.is_running = False
        self.is_closed = False

    def _create_workers(self, env_creator, worker_cls):
        """
        Creates self.num_workers worker processes.
        Each worker receives an approximate equal share of emulators and corresponding variables.
        All Workers run concurrently but each of them updates their emulators sequentially.
        :return: A list of created workers
        """
        if self.num_emulators < self.num_workers:
            raise BatchEmulatorError(
                "{}: Number of emulators must be bigger than number of workers".format(type(self).__name__)
            )

        workers = [None]*self.num_workers
        # emulator_slices[i] will store number of emulators being processed by the i-th worker:
        emulator_slices = [None]* self.num_workers
        #following segment creates workers and splits emulators between them as fairly as possible:
        min_local_ems = self.num_emulators // self.num_workers
        extra_ems = self.num_emulators % self.num_workers
        l = r = 0

        for wid in range(self.num_workers):
            num_ems = min_local_ems + int(wid < extra_ems)
            l, r = r, min(r+num_ems, self.num_emulators)

            emulator_slices[wid] = (l,r) #num_ems

            create_envs = lambda l=l,r=r: \
                [env_creator.create_environment(i) for i in range(l,r)]

            workers[wid] = worker_cls(
                wid, create_envs, self.worker_conns[wid], self.conns[wid],
            )

        return workers, emulator_slices

    def start_workers(self):
        """
        Starts worker processes.
        Despite start_workers resembles the reset method from emulator classes,
        it doesn't reset emulators to new episodes.
        Ideally, start_worker should be called only once before the training stage.
        """

        if self.is_closed:
            raise BatchEmulatorError('{} is already closed'.format(type(self)))
        if not self.is_running:
            self.is_running = True
            for r in self.workers:
                assert r.daemon == True, 'Processes should be run as daemons!'
                r.daemon = True #already asserted in the __init__
                r.start()
            # each proccess should send or receive only from one end of a pipe:
            for conn in self.worker_conns:
                conn.close()

    def stop_workers(self):
        """Just use close()"""
        # why i ever need this function?
        if self.is_closed:
            raise BatchEmulatorError('{} is already closed'.format(type(self)))
        if self.is_running and not self.is_closed:
            for conn in self.conns:
                conn.send((self._command.CLOSE, None))
            self.is_running = False

    def close(self):
        """Stops worker processes and joins them."""
        if not self.is_closed:
            self.stop_workers()
            for worker in self.workers:
                worker.join()
            self.is_closed = True

    def next(self, action):
        """
        Performs given actions on the corresponding emulators i.e. performs action[i] on emulator[i].
        :param action:  Array of actions. if action space is discrete one-hot encoding is used.
        :return: states, rewards, dones, infos
        """
        for conn, (l,r) in zip(self.conns, self.emulators_slices):
            conn.send((self._command.NEXT, action[l:r]))

        results = [conn.recv() for conn in self.conns]
        obs, rs, dones, info_list =  zip(*results)

        return obs, rs, dones, self._make_infos(info_list)

    def reset_all(self):
        for conn in self.conns:
            conn.send((self._command.RESET, None))

        results = [conn.recv() for conn in self.conns]
        obs, info_list = zip(*results)

        return obs, self._make_infos(info_list)

    def _make_infos(self, info_list):
        infos = {k:[] for k in info_list[0].keys()}
        for info in info_list:
            for k, v in info.items():
                infos[k].append(v)
        return infos

    def call_method(self, method_name: str, method_args: List[Tuple[list, dict]] = None) -> Any:
        """
        Calls an arbitrary method for all running emulators
        This call_method works slower than next and reset_all methods as it doesn't make use of arrays in shared memory.
        Use this if you need to interact with the emulated environments in some unusual manner.
        But try to call this method as rare as possible if you don't want to slower your learning fps.
        :arg method_name: name of the method you want to call
        :arg method_args: list of shape [(emulator1_args, emulator1_kwargs),(emulator2_args, emulator2_kwargs), ...]
        """
        cmd = self._command.CALL_METHOD

        for conn, (l, r) in enumerate(self.conns, self.emulators_slices):
            data = {'arg_list:':method_args[l:r], 'method_name':method_name}
            conn.send((cmd, data))

        results = [conn.recv() for conn in self.conns]
        return zip(*results)


class SequentialBatchEnv(BaseBatchEnv):
    """
    SequentialBatchEnv creates num_emulators environments and updates them one by one.
    It doesn't use multiprocessing.
    SequentialBatchEnv is mainly used for testing and evaluation of already trained network.
    """
    def __init__(self, env_creator, num_emulators,
                 auto_reset=True, extra_vars='all', init_env_id=1000,
                 specific_emulator_args=None):
        super(SequentialBatchEnv, self).__init__(env_creator, num_emulators)
        inputs, outputs, extra_outputs = self._create_variables(env_creator, extra_vars)
        for k, var in inputs.items(): setattr(self, k, var)
        for k, var in outputs.items(): setattr(self, k, var)
        self.info = {k:var for k,var in extra_outputs.items()}
        self.auto_reset = auto_reset
        self.completed = [False]*num_emulators
        em_args = specific_emulator_args if specific_emulator_args else {}
        self.emulators = [env_creator.create_environment(i+init_env_id,**em_args) for i in range(num_emulators)]

    def reset_all(self):
        """
        Starts new episodes in all emulators.
        :return:
           An array of emulators' states,
           a dict with extra variables(if there is no such variables then the dict is empty).
        """
        for i, em in enumerate(self.emulators):
            self.state[i], info = em.reset()
            self.completed[i] = False
            for k in self.info:
                self.info[k][i] = info[k]
        return self.state, self.info

    def next(self, action):
        """
        Sequentially performs action on each corresponding emulator, i.e. performs action[i] on emulator[i].
        :param action: Array of actions. if action space is discrete one-hot encoding is used.
        :return: states, rewards, dones, infos
        """
        self.action[:] = action
        for i, (em, act) in enumerate(zip(self.emulators, self.action)):
            if not self.completed[i]:
                new_s, self.reward[i], self.is_done[i], info = em.next(act)
                if self.is_done[i] and self.auto_reset:
                    new_s, info = em.reset()
                elif self.is_done[i] and not self.auto_reset:
                    new_s = 0
                    #for k in info: info[k] = 0
                    self.completed[i] = True

                self.state[i] = new_s
                for k in self.info:
                    self.info[k][i] = info[k]
            else:
                self.reward[i] = 0
                self.is_done[i] = True

        return self.state, self.reward, self.is_done, self.info

    def call_method(self, method_name: str, method_args: List[Tuple[list, dict]] = None) -> List[Any]:
        results = []
        for em, (args, kwargs) in zip(self.emulators, method_args):
            em_result = getattr(em, method_name)(*args, **kwargs)
            results.append(em_result)
        return results

    def close(self):
        for em in self.emulators:
            em.close()