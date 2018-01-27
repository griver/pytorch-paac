import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double


class Runners(object):

    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, EmulatorRunner, emulators, workers, variables):
        self.variables = [self._get_shared(var) for var in variables]
        self.workers = workers
        self.queues = [Queue() for _ in range(workers)]
        self.barrier = Queue()
        # emulators_for_worker[i] contains emulators
        # which will be used by the ith worker.
        # Same logic applies for vars_for_worker and emulators_and_vars_for_worker
        emulators_for_worker = np.split(emulators, workers)
        vars_for_worker = zip(*[np.split(var, workers) for var in self.variables])
        emulators_and_vars_for_worker = zip(emulators_for_worker, vars_for_worker)
        
        self.runners = []
        for i, (emulators, vars) in enumerate(emulators_and_vars_for_worker):
            runner = EmulatorRunner(i, emulators, vars, self.queues[i], self.barrier)
            self.runners.append(runner)

        #The original runners definition. It is shorter but harder for reading:
        #self.runners = [EmulatorRunner(i, emulators, vars, self.queues[i], self.barrier) for i, (emulators, vars) in
        #                enumerate(zip(np.split(emulators, workers), zip(*[np.split(var, workers) for var in self.variables])))]


    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """
        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for r in self.runners:
            r.start()

    def stop(self):
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.variables

    def update_environments(self):
        for queue in self.queues:
            queue.put(True)

    def wait_updated(self):
        for wd in range(self.workers):
            self.barrier.get()
