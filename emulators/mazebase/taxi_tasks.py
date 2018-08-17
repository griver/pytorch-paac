from enum import IntEnum
import numpy as np

class TaskStatus(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    FAIL = 2

class TaxiTask(object):
    task_id = -1

    def __init__(self,
                 duration=10000,
                 status=TaskStatus.RUNNING,
                 finish_action=None):
        super(TaxiTask, self).__init__()
        self.status = status
        self.step = 0
        self.duration = duration
        self.finish_action=finish_action

    def allowed_action(self, action: str) -> bool:
        return True

    def update_status(self, state_resume) -> TaskStatus:
        raise NotImplementedError()

    @classmethod
    def is_available(Class, state_resume):
        raise NotImplementedError()

    @classmethod
    def create(Class, state_resume, **kwargs):
        raise NotImplementedError()

    def completed(self):
        return self.status == TaskStatus.SUCCESS

    def failed(self):
        return self.status == TaskStatus.FAIL

    def running(self):
        return self.status == TaskStatus.RUNNING

    def as_info_dict(self):
        return {
            'task_id':self.task_id,
            'task_status':int(self.status)
        }

    def __str__(self):
        task_name = type(self).__name__
        return "{}: status={}, t={}, lim={}".format(
            task_name, self.status.name, self.step, self.duration)


class FullTaxi(TaxiTask):
    task_id = 0

    def __init__(self, *args, **kwargs):
        super(FullTaxi,self).__init__(*args, **kwargs)

    @classmethod
    def is_available(Class, state_resume):
        return not state_resume.passenger_in_taxi

    @classmethod
    def create(Class, state_resume, **kwargs):
        return Class(**kwargs)

    def update_status(self, state_resume):
        self.step += 1
        finish_condition = (not state_resume.passenger_in_taxi) \
                           and (state_resume.loc_passenger == state_resume.loc_destination)
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status


class PickUp(TaxiTask):
    task_id = 1

    def __init__(self, init_loc, *args, **kwargs):
        super(PickUp,self).__init__(*args, **kwargs)
        self.init_loc = init_loc

    @classmethod
    def is_available(Class, state_resume):

        return (not state_resume.passenger_in_taxi) and \
               (state_resume.loc_passenger == state_resume.loc_taxi)

    @classmethod
    def create(Class, state_resume, **kwargs):
        init_loc = state_resume.loc_taxi
        kwargs.setdefault('duration', 5)
        return Class(init_loc, **kwargs)

    def update_status(self, state_resume)-> TaskStatus:
        self.step += 1

        finish_condition = state_resume.passenger_in_taxi
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status


class ConveyPassenger(TaxiTask):
    task_id = 2

    def __init__(self, *args, **kwargs):
        super(ConveyPassenger,self).__init__(*args,**kwargs)

    @classmethod
    def is_available(Class, state_resume):
        return state_resume.passenger_in_taxi #and\
               #state_resume.loc_taxi != state_resume.loc_destination

    @classmethod
    def create(Class, state_resume, **kwargs):
        kwargs.setdefault('duration',100)
        return Class(**kwargs)

    def update_status(self, state_resume)-> TaskStatus:
        self.step += 1

        finish_condition = (not state_resume.passenger_in_taxi) and\
                           (state_resume.loc_destination == state_resume.loc_passenger)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status


class FindPassenger(TaxiTask):
    task_id = 3

    def __init__(self, passenger_in_taxi, *args,**kwargs):
        super(FindPassenger, self).__init__(*args, **kwargs)
        self.passenger_in_taxi = passenger_in_taxi

    @classmethod
    def is_available(Class, state_resume):
        return state_resume.passenger_in_taxi is False \
               and state_resume.loc_passenger != state_resume.loc_taxi

    @classmethod
    def create(Class, state_resume, **kwargs):
        in_taxi = state_resume.passenger_in_taxi
        kwargs.setdefault('duration', 200)
        return Class(in_taxi, **kwargs)

    def allowed_action(self, action):
        return action not in ('pickup','dropoff')

    def update_status(self, state_resume)-> TaskStatus:
        self.step += 1
        loc_p = state_resume.loc_passenger
        loc_t = state_resume.loc_taxi

        finish_condition = (loc_t == loc_p) and (not state_resume.passenger_in_taxi)
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status

class ReachDestination(TaxiTask):
    task_id = 4

    def __init__(self, passenger_in_taxi, *args, **kwargs):
        super(ReachDestination, self).__init__(*args, **kwargs)
        self.passenger_in_taxi = passenger_in_taxi

    @classmethod
    def is_available(Class, state_resume):
        return state_resume.loc_destination != state_resume.loc_taxi

    @classmethod
    def create(Class, state_resume, **kwargs):
        in_taxi = state_resume.passenger_in_taxi
        kwargs.setdefault('duration', 200)
        return Class(in_taxi, **kwargs)

    def allowed_action(self, action):
        return action not in ('pickup','dropoff')

    def update_status(self, state_resume)-> TaskStatus:
        self.step += 1

        finish_condition = (state_resume.loc_taxi == state_resume.loc_destination) \
                           and (state_resume.passenger_in_taxi == self.passenger_in_taxi)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status


class DropOff(TaxiTask):
    task_id = 5

    def __init__(self, init_loc, *args, **kwargs):
        super(DropOff,self).__init__(*args,**kwargs)
        self.init_loc = init_loc

    @classmethod
    def is_available(Class, state_resume):
        return state_resume.passenger_in_taxi

    @classmethod
    def create(Class, state_resume, **kwargs):
        init_loc = state_resume.loc_taxi
        kwargs.setdefault('duration',5)
        return Class(init_loc, **kwargs)

    def update_status(self, state_resume)-> TaskStatus:
        self.step += 1

        finish_condition = (not state_resume.passenger_in_taxi) and\
                           (self.init_loc == state_resume.loc_passenger)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == state_resume.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

tasks_dict = dict(
    pickup=PickUp,
    dropoff=DropOff,
    find_p=FindPassenger,
    convey_p=ConveyPassenger,
    reach_d=ReachDestination,
    full_taxi=FullTaxi
)

class AbstractTaskManager(object):

    def next(self, *args, **kwargs):
        raise NotImplementedError('This method is not implemented yet!')

    def required_state_vars(self):
        raise NotImplementedError('This method is not implemented yet!')

class TaskManager(AbstractTaskManager):
    """
    Creates a new task available in the current game state.
    The task will belong to one of the task types passed
    to the TaskManager instance.
    """

    def __init__(self, tasks, priorities=None, extra_task_kwargs=None):
        """
        :param task_types: A list of task classes to choose(a next task) from
        :param priorities: a type priority affects the probability that a next task would be an instance of this type.
        """
        self.extra_task_kwargs = extra_task_kwargs if extra_task_kwargs else {}

        tasks = self.__process_tasks(tasks)
        self._task_types = np.asarray(tasks)
        if not priorities:
            num_tasks = len(tasks)
            self._priorities = np.full(len(tasks), 1./num_tasks)
        else:
            self._priorities = np.asarray(priorities)/sum(priorities)

    def __process_tasks(self, tasks):
        if not len(tasks):
            raise ValueError('TaskManager:You should specify at least one task!')
        if not all(isinstance(t, TaxiTask) for t in tasks):
            try:
                tasks = [tasks_dict[t] for t in tasks]
            except KeyError as k:
                raise ValueError('tasks should be a list of subclasses of TaxiTask '
                                 'or a list of names present in the dictionary'
                                 ' emulator.mazebase.taxi_task.tasks_dict')
        return tasks

    def next(self, state_resume, rnd_state):
        is_available = [t.is_available(state_resume) for t in self._task_types]
        priorities = self._priorities[is_available]
        task_types = self._task_types[is_available]
        #print('available tasks:', [t.__name__ for t in task_types])
        selected_task_type = rnd_state.choice(task_types,
                                              p=priorities/sum(priorities))
        task = selected_task_type.create(state_resume, **self.extra_task_kwargs)
        #print('selected task:', task, 'as_dict:', task.as_info_dict())
        return task

    def required_state_vars(self):
        all_required_vars = set()
        all_vars = [task_cls.required_state_vars for task_cls in self._task_types]
        all_required_vars.update(*all_vars)
        return all_required_vars