from multiprocessing import Process
from itertools import chain
from enum import IntEnum
import logging

class WorkerError(Exception): pass


class PipeWorker(Process):
    class Command(IntEnum):
        CLOSE = 0
        NEXT = 1
        RESET = 2
        CALL_METHOD = 3

    def __init__(self, id, create_envs, worker_conn, master_conn):
        super(PipeWorker, self).__init__()
        self.daemon = True
        self.id = id
        self.conn = worker_conn
        self.master_conn = master_conn
        self.create_envs = create_envs

    def run(self):
        super(PipeWorker, self).run()
        self._run()

    def _run(self):
        self.master_conn.close()
        del self.master_conn

        envs = self.create_envs()
        num_envs = len(envs)
        states = [None]*num_envs
        infos = [None]*num_envs
        rs = [None]*num_envs
        dones = [None]*num_envs

        try:
            while True:
                command, data = self.conn.recv()

                if command == self.Command.NEXT:
                    for i, (env, action) in enumerate(zip(envs, data)):
                        states[i], rs[i], dones[i], infos[i] = env.next(action)
                        if dones:
                            states[i], infos[i] = env.reset()
                    self.conn.send((states, rs, dones, infos))

                elif command == self.Command.CALL_METHOD:
                    results = self._call_method(envs, *data)
                    self.conn.send(results)

                elif command == self.Command.RESET:
                    for i, env in enumerate(envs):
                        states[i], infos[i] = env.reset()
                    self.conn.send((states, infos))

                elif command == self.Command.CLOSE:
                    self.conn.close()
                    break

                else:
                    raise WorkerError("{} has received unknown command {}".format(type(self),command))
        except KeyboardInterrupt:
            logging.exception('{} caught KeyboardInterrupt!'.format(type(self)))

        finally:
            for env in envs:
                env.close()
            logging.info('{}#{} finished!'.format(type(self),self.id+1))

    def _call_method(self, envs, method_name, arg_list):
        results = [None]*len(envs)
        for i, emulator in enumerate(envs):
            args, kwargs = arg_list[i]
            results[i] = getattr(emulator, method_name)(*args, **kwargs)
        return results


class SharedMemWorker(Process):
    required_outputs = {'state', 'is_done', 'reward'}
    required_inputs = {'action'}
    class Command(IntEnum):
        CLOSE = 0
        NEXT = 1
        RESET = 2
        CALL_METHOD = 3

    def __init__(self, id, create_emulators, queue, barrier, required_vars, extra_outputs):
        super(SharedMemWorker, self).__init__()
        self.daemon = True
        self.id = id
        self.create_emulators = create_emulators
        self.queue = queue
        self.barrier = barrier
        self.__check_variables(required_vars)
        self._init_required(required_vars) #sets them as instance attributes
        self.info = {k:v for k,v in extra_outputs.items()} #lets create a shallow copy, just in case.

    def __check_variables(self, given):
        error_msg = "{} requires a shared variable {} to store" +\
                    "data essential for interaction with the game emulators."
        for name in chain(self.required_outputs, self.required_inputs):
            if name not in given:
                raise WorkerError(error_msg.format(type(self), name))

    def _init_required(self, variables):
        for name in chain(self.required_inputs, self.required_outputs):
            setattr(self, name, variables[name])

    def run(self):
        super(SharedMemWorker, self).run()
        self._run()

    def _run(self):
        """
        Creates emulators, then in the cycle it waits for a command from an algorithm.
        The received command is sequentially performed on each of the emulators.
        If it gets some unknown command WorkerError is raised and the process is terminated.
        """
        emulators = self.create_emulators()
        try:
            while True:
                command, data = self.queue.get()

                if command == self.Command.NEXT:
                    for i, (emulator, action) in enumerate(zip(emulators, self.action)):
                        new_s, reward, is_done, info = emulator.next(action)
                        if is_done:
                            self.state[i], info = emulator.reset()
                        else: #so we never return terminal states
                            self.state[i] = new_s
                        self.reward[i] = reward
                        self.is_done[i] = is_done
                        for k in self.info:
                            self.info[k][i] = info[k]
                    self.barrier.put(True)

                elif command == self.Command.CALL_METHOD:
                    self._call_method(emulators, **data)

                elif command == self.Command.RESET:
                    for i, emulator in enumerate(emulators):
                        self.state[i], info = emulator.reset()
                        for k in self.info:
                            self.info[k][i] = info[k]
                    self.barrier.put(True)

                elif command == self.Command.CLOSE:
                    break

                else:
                    raise WorkerError("{} has received unknown command {}".format(type(self),command))
        finally:
            for emulator in emulators: emulator.close()
            logging.debug('SharedMemWorker#{} finished!'.format(self.id+1))

    def _call_method(self, emulators, method_name, arg_list):
        results = [None]*len(emulators)
        for i, emulator in enumerate(emulators):
            args, kwargs = arg_list[i]
            results[i] = getattr(emulator, method_name)(*args, **kwargs)
        self.barrier.put(results)