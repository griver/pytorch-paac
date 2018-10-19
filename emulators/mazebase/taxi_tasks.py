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

    def update_status(self, game) -> TaskStatus:
        raise NotImplementedError()

    @classmethod
    def is_available(Class, game):
        raise NotImplementedError()

    @classmethod
    def create(Class, game, **kwargs):
        raise NotImplementedError()

    def completed(self):
        return self.status == TaskStatus.SUCCESS

    def failed(self):
        return self.status == TaskStatus.FAIL

    def running(self):
        return self.status == TaskStatus.RUNNING

    def as_info_dict(self):
        return {
            'name': type(self).__name__,
            'id':self.task_id,
            'status':int(self.status),
            'num_steps':self.step
        }

    def __str__(self):
        task_name = type(self).__name__
        return "{}: status={}, t={}, lim={}".format(
            task_name, self.status.name, self.step, self.duration)

    def min_steps_to_complete(self, game):
        raise NotImplementedError()


class FullTaxi(TaxiTask):
    task_id = 0

    def __init__(self, *args, **kwargs):
        super(FullTaxi,self).__init__(*args, **kwargs)

    @classmethod
    def is_available(Class, game):
        return not game.passenger.is_pickedup

    @classmethod
    def create(Class, game, **kwargs):
        return Class(**kwargs)

    def update_status(self, game):
        self.step += 1
        finish_condition = (not game.passenger.is_pickedup) \
                           and (game.passenger.location == game.target.location)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        p_loc = game.passenger.location
        a_loc = game.agent.location
        t_loc = game.target.location
        passenger_in_taxi = game.passenger.is_pickedup
        if passenger_in_taxi:
            return 1 + game.distance(a_loc, t_loc)
        elif p_loc == t_loc:
            return 0
        else:
            return 2 + game.distance(a_loc, p_loc) + game.distance(p_loc, t_loc)


class PickUpPassenger(TaxiTask):
    task_id = 1

    def __init__(self, init_loc, *args, **kwargs):
        super(PickUpPassenger, self).__init__(*args, **kwargs)
        self.init_loc = init_loc

    @classmethod
    def is_available(Class, game):

        return (not game.passenger.is_pickedup) and \
               (game.passenger.location == game.agent.location)

    @classmethod
    def create(Class, game, **kwargs):
        init_loc = game.agent.location
        kwargs.setdefault('duration', 60)
        return Class(init_loc, **kwargs)

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = game.passenger.is_pickedup
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        passenger_in_taxi = game.passenger.is_pickedup
        if passenger_in_taxi:
            return 0
        else:
            return 1 + game.distance(game.agent.location, game.passenger.location)


class ConveyPassenger(TaxiTask):
    task_id = 2

    def __init__(self, *args, **kwargs):
        super(ConveyPassenger,self).__init__(*args,**kwargs)

    @classmethod
    def is_available(Class, game):
        return game.passenger.is_pickedup

    @classmethod
    def create(Class, game, **kwargs):
        kwargs.setdefault('duration',200)
        return Class(**kwargs)

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = (not game.passenger.is_pickedup) and\
                           (game.target.location == game.passenger.location)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        p_loc = game.passenger.location
        a_loc = game.agent.location
        t_loc = game.target.location
        passenger_in_taxi = game.passenger.is_pickedup
        if passenger_in_taxi:
            return 1 + game.distance(a_loc, t_loc)
        elif p_loc == t_loc:
            return 0
        else:
            return 2 + game.distance(a_loc, p_loc) + game.distance(p_loc, t_loc)


class FindPassenger(TaxiTask):
    task_id = 3

    def __init__(self, passenger_in_taxi, *args,**kwargs):
        super(FindPassenger, self).__init__(*args, **kwargs)
        self.passenger_in_taxi = passenger_in_taxi

    @classmethod
    def is_available(Class, game):
        agent_loc = game.agent.location
        #check if it makes sense to find passenger:
        return agent_loc != game.passenger.location

    @classmethod
    def create(Class, game, **kwargs):
        in_taxi = game.passenger.is_pickedup
        kwargs.setdefault('duration', 200)
        return Class(in_taxi, **kwargs)

    def allowed_action(self, action):
        return action not in ('pickup','dropoff')

    def update_status(self, game)-> TaskStatus:
        self.step += 1
        loc_p = game.passenger.location
        loc_t = game.agent.location

        finish_condition = (loc_t == loc_p) and (not game.passenger.is_pickedup)
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status

    def min_steps_to_complete(self, game):
        return game.distance(game.agent.location, game.passenger.location)


class MoveUp(TaxiTask):
    task_id = 4

    def __init__(self, agent_loc, *args, **kwargs):
        super(MoveUp, self).__init__(*args, **kwargs)
        self.agent_loc = agent_loc

    @classmethod
    def is_available(Class, game):
        x,y = game.agent.location
        return (y+1) < game.height

    @classmethod
    def create(Class, game, **kwargs):
        agent_loc = game.agent.location
        kwargs.setdefault('duration', 60)
        return Class(agent_loc, **kwargs)

    def update_status(self, game):
        self.step += 1
        x,y = game.agent.location
        finish_condition = (y + 1) >= game.height
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status


class MoveDown(TaxiTask):
    task_id = 5

    def __init__(self, agent_loc, *args, **kwargs):
        super(MoveDown, self).__init__(*args, **kwargs)
        self.agent_loc = agent_loc

    @classmethod
    def is_available(Class, game):
        x, y = game.agent.location
        return y > 0

    @classmethod
    def create(Class, game, **kwargs):
        agent_loc = game.agent.location
        kwargs.setdefault('duration', 60)
        return Class(agent_loc, **kwargs)

    def update_status(self, game):
        self.step += 1
        x, y = game.agent.location
        finish_condition = (y == 0)
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status


class ReachDestination(TaxiTask):
    task_id = 6

    def __init__(self, *args, **kwargs):
        super(ReachDestination, self).__init__(*args, **kwargs)

    @classmethod
    def is_available(Class, game):
        return game.target.location != game.agent.location

    @classmethod
    def create(Class, game, **kwargs):
        kwargs.setdefault('duration', 200)
        return Class(**kwargs)

    def allowed_action(self, action):
        return action not in ('pickup','dropoff')

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = (game.agent.location == game.target.location)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status

    def min_steps_to_complete(self, game):
        return game.distance(game.agent.location, game.target.location)


class DropOffPassenger(TaxiTask):
    task_id = 7

    def __init__(self, init_loc, *args, **kwargs):
        super(DropOffPassenger, self).__init__(*args, **kwargs)
        self.init_loc = init_loc

    @classmethod
    def is_available(Class, game):
        return game.passenger.is_pickedup

    @classmethod
    def create(Class, game, **kwargs):
        init_loc = game.agent.location
        kwargs.setdefault('duration',60)
        return Class(init_loc, **kwargs)

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = (not game.passenger.is_pickedup) and\
                           (self.init_loc == game.passenger.location)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        p_loc = game.passenger.location
        a_loc = game.agent.location
        passenger_in_taxi = game.passenger.is_pickedup
        if passenger_in_taxi:
            return 1 + game.distance(a_loc, self.init_loc)
        elif p_loc == self.init_loc:
            return 0
        else:
            return 2 + game.distance(a_loc, p_loc) + game.distance(p_loc, self.init_loc)


class FindCargo(TaxiTask):
    task_id = 8

    def __init__(self, cargo_in_taxi, *args,**kwargs):
        super(FindCargo, self).__init__(*args, **kwargs)
        self.cargo_in_taxi = cargo_in_taxi

    @classmethod
    def is_available(Class, game):
        agent_loc = game.agent.location
        #check if it makes sense to find the cargo:
        return agent_loc != game.cargo.location

    @classmethod
    def create(Class, game, **kwargs):
        in_taxi = game.cargo.is_pickedup
        kwargs.setdefault('duration', 200)
        return Class(in_taxi, **kwargs)

    def allowed_action(self, action):
        return action not in ('pickup','dropoff')

    def update_status(self, game)-> TaskStatus:
        self.step += 1
        loc_p = game.cargo.location
        loc_t = game.agent.location

        finish_condition = (loc_t == loc_p) and (not game.cargo.is_pickedup)
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING

        return self.status

    def min_steps_to_complete(self, game):
        return game.distance(game.agent.location, game.cargo.location)



class PickUpCargo(TaxiTask):
    task_id = 9

    def __init__(self, init_loc, *args, **kwargs):
        super(PickUpCargo, self).__init__(*args, **kwargs)
        self.init_loc = init_loc

    @classmethod
    def is_available(Class, game):

        return (not game.cargo.is_pickedup) and \
               (game.cargo.location == game.agent.location)

    @classmethod
    def create(Class, game, **kwargs):
        init_loc = game.agent.location
        kwargs.setdefault('duration', 60)
        return Class(init_loc, **kwargs)

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = game.cargo.is_pickedup
        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        cargo_in_taxi = game.cargo.is_pickedup
        if cargo_in_taxi:
            return 0
        else:
            return 1 + game.distance(game.agent.location, game.cargo.location)


class ConveyCargo(TaxiTask):
    task_id = 10

    def __init__(self, *args, **kwargs):
        super(ConveyCargo,self).__init__(*args,**kwargs)

    @classmethod
    def is_available(Class, game):
        return game.cargo.is_pickedup #and\
               #game.agent.location != game.target.location

    @classmethod
    def create(Class, game, **kwargs):
        kwargs.setdefault('duration',200)
        return Class(**kwargs)

    def update_status(self, game)-> TaskStatus:
        self.step += 1

        finish_condition = (not game.cargo.is_pickedup) and\
                           (game.target.location == game.cargo.location)

        finish_act = (self.finish_action is None) \
                     or (self.finish_action == game.agent.last_performed_act)

        if finish_condition and finish_act:
            self.status = TaskStatus.SUCCESS
        elif self.step >= self.duration:
            self.status = TaskStatus.FAIL
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def min_steps_to_complete(self, game):
        c_loc = game.cargo.location
        a_loc = game.agent.location
        t_loc = game.target.location
        cargo_in_taxi = game.cargo.is_pickedup
        if cargo_in_taxi:
            return 1 + game.distance(a_loc, t_loc)
        elif c_loc == t_loc:
            return 0
        else:
            return 2 + game.distance(a_loc, c_loc) + game.distance(c_loc, t_loc)


tasks_dict = dict(
    pickup=PickUpPassenger,
    dropoff=DropOffPassenger,
    find_p=FindPassenger,
    convey_p=ConveyPassenger,
    reach_d=ReachDestination,
    move_up=MoveUp,
    move_down=MoveDown,
    full_taxi=FullTaxi,
    find_c=FindCargo,
    pickup_c=PickUpCargo,
    convey_c=ConveyCargo,
)


class AbstractTaskManager(object):

    def next(self, *args, **kwargs):
        raise NotImplementedError('This method is not implemented yet!')

    def required_state_vars(self):
        raise NotImplementedError('This method is not implemented yet!')


class TaskManagerError(Exception):
    pass


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

    def next(self, game, rnd_state):
        is_available = [t.is_available(game) for t in self._task_types]
        priorities = self._priorities[is_available]
        task_types = self._task_types[is_available]
        #print('available tasks:', [t.__name__ for t in task_types])
        selected_task_type = rnd_state.choice(task_types,
                                              p=priorities/sum(priorities))
        task = selected_task_type.create(game, **self.extra_task_kwargs)
        #print('selected task:', task, 'as_dict:', task.as_info_dict())
        return task

    def required_state_vars(self):
        all_required_vars = set()
        all_vars = [task_cls.required_state_vars for task_cls in self._task_types]
        all_required_vars.update(*all_vars)
        return all_required_vars


class LifelongTaskManager(TaskManager):
    """
    Honestly, LifelongTaskManager is a fast workaround, for the task scheduling in our lifelong experiments.
    LifelongTaskManager splits task in two lists: cargo-related tasks and passenger-releated tasks.
    And tries to call tasks in one list sequentially. If this can't be done, it randomly selects the new list of tasks and
    starts anew.
    """

    def __init__(self, tasks, *args, **kwargs):
        super(LifelongTaskManager, self).__init__(tasks, *args, **kwargs)
        self.task_seqs = self._generate_task_sequences(self._task_types)
        self.current_seq = None
        self.prev_task_id = None

    def _generate_task_sequences(self, given_tasks):
        given_tasks = set(given_tasks)

        cargo_seq = [FindCargo, PickUpCargo, ConveyCargo]
        passenger_seq = [FindPassenger, PickUpPassenger, ConveyPassenger]

        extra_tasks = given_tasks.difference(cargo_seq).difference(passenger_seq)
        assert len(extra_tasks) == 0, "LifelongTaskManager is not suited for the tasks: {}".format(extra_tasks)

        cargo_seq = [t for t in cargo_seq if t in given_tasks]
        passenger_seq = [t for t in passenger_seq if t in given_tasks]
        return dict(cargo_seq=cargo_seq, passenger_seq=passenger_seq)


    def select_from_group(self, group_name, game, rnd_state):
        next_group = group_name

        tasks = self.task_seqs[group_name]
        is_available = [t.is_available(game) for t in tasks]
        if any(is_available):
            indices = np.arange(len(tasks))
            indices = indices[is_available]
            idx = rnd_state.choice(indices)
            task = tasks[idx]
            #no need to assign group if this is a final subtask.
            if task == tasks[-1]: next_group = None
        else:
            task = None
            next_group = None

        return task, next_group

    def next(self, game, rnd_state):

        if self.current_seq is not None:
            task, next_seq = self.select_from_group(self.current_seq, game, rnd_state)
            if task:
                self.current_seq = next_seq
                return task.create(game, **self.extra_task_kwargs)


        for seq in rnd_state.permutation(sorted(self.task_seqs.keys())):
            task, next_seq = self.select_from_group(seq, game, rnd_state)
            if task:
                self.current_seq = next_seq
                return task.create(game, **self.extra_task_kwargs)
        else:
            task_names = {k:[el.__name__ for el in v] for k,v in self.task_seqs.items()}
            msg = "Can't find the next available task({0}) at the state:" \
                  " passenger_loc={1.location}, cargo_loc={2.location} agent_loc={3.location}," \
                  " target_loc={4.location} obj_in_taxi = {5}".format(
                task_names, game.passenger, game.cargo, game.agent, game.target,
                type(game.agent.item).__name__ if game.agent.item else None
            )
            raise TaskManagerError(msg)

    def old_next(self, game, rnd_state):

        prev_task = game._tasks_history[-1]['name'] if game._tasks_history else ''

        related_tasks = [self._related_tasks(prev_task, t.__name__) for t in self._task_types]
        available_tasks = [t.is_available(game) for t in self._task_types]
        available_and_related_tasks = np.logical_and(available_tasks, related_tasks)

        not_finish_task = ('Dropoff' not in prev_task) and ('Convey' not in prev_task)
        if not_finish_task and any(available_and_related_tasks):
            candidate_mask = available_and_related_tasks
        else:
            candidate_mask = available_tasks

        priorities = self._priorities[candidate_mask]
        task_types = self._task_types[candidate_mask]

        selected_task_type = rnd_state.choice(task_types,
                                              p=priorities/sum(priorities))
        task = selected_task_type.create(game, **self.extra_task_kwargs)

        return task

    def old_next2(self, game, rnd_state):
        """
        Select next avaialble task while preventing
        tasks ReachPassenger and ReachCargo looping between each other
        """
        def is_find_task(name_or_type):
            name = name_or_type if type(name_or_type) is str else name_or_type.__name__
            return "Find" in name

        prev_task = game._tasks_history[-1]['name'] if game._tasks_history else ''

        available_tasks = [t.is_available(game) for t in self._task_types]
        if is_find_task(prev_task):
            not_find_task = [not is_find_task(t) for t in self._task_types]
            available_and_not_reach = np.logical_and(available_tasks, not_find_task)
            if any(available_and_not_reach):
                available_tasks = available_and_not_reach

        priorities = self._priorities[available_tasks]
        task_types = self._task_types[available_tasks]

        selected_task_type = rnd_state.choice(task_types,
                                              p=priorities / sum(priorities))
        task = selected_task_type.create(game, **self.extra_task_kwargs)

        return task


class TaskStats(object):
    """Stores information about agent's success rate in completing subtasks."""
    def __init__(self):
        self._stats = {}

    def add_task_history(self, task_history):
        for task_info in task_history:
            t_name = task_info['name']
            t_data = self._stats.setdefault(t_name, {
                'status':[],
                'num_steps':[]
            })
            t_data['num_steps'].append(task_info['num_steps'])
            t_data['status'].append(task_info['status'])

    def __str__(self):
        lines = []
        for k in sorted(self._stats):
            data = self._stats[k]
            statuses = np.array(data['status']) == TaskStatus.SUCCESS
            success = statuses.mean() * 100
            mean_steps = np.mean(data['num_steps'])
            num_tasks = len(statuses)
            lines.append(
                "{}({}): success={:.1f}% num_steps={:.1f}".format(
                    k,num_tasks,success, mean_steps)
            )
        return '\n'.join(lines)

    def logging_form(self):
        logging_dict = {}
        for k in sorted(self._stats):
            data = self._stats[k]
            statuses = np.array(data['status']) == TaskStatus.SUCCESS
            success = statuses.mean() * 100
            mean_steps = np.mean(data['num_steps'])
            logging_dict[k] = {'success':success, 'mean_steps':mean_steps}
        return logging_dict