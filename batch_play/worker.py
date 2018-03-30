from multiprocessing import Process
from itertools import chain
from enum import IntEnum
import logging

class WorkerError(Exception): pass

class WorkerProcess(Process):
    required_outputs = {'state', 'is_done', 'reward'}
    required_inputs = {'action'}
    class Command(IntEnum):
        CLOSE = 0
        NEXT = 1
        RESET = 2

    def __init__(self, id, create_emulators, queue, barrier, required_vars, extra_outputs):
        super(WorkerProcess, self).__init__()
        self.daemon = True
        self.id = id
        self.create_emulators = create_emulators
        self.queue = queue
        self.barrier = barrier
        self.__check_variables(required_vars)
        self._init_required(required_vars) #sets them as instance attributes
        self.info = {k:v for k,v in extra_outputs} #lets create a shallow copy, just in case.

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
        super(WorkerProcess, self).run()
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
                command = self.queue.get()
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
            logging.debug('WorkerProcess#{} finished!'.format(self.id+1))

