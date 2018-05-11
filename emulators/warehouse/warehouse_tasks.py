from enum import IntEnum
import numpy as np
import itertools as it
import pandas as pd
import warnings

class TaskStatus(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    FAIL = 2


class WarehouseTask(object):
    #vizdoom state variables are ints or fixed points
    none_value = -1
    penalty = -0.005
    strong_penalty = -0.02
    fail_penalty = -1.
    goal_reward = 1.
    duration = float('inf')
    required_state_vars = set()
    task_id = 0

    def __init__(self, duration=None, info_dict=None):
        self.n_steps = 0
        self.status = TaskStatus.RUNNING
        if duration:
            self.duration = duration
        self._info_dict = info_dict

    @classmethod
    def is_available(Class, state_info):
        raise NotImplementedError()

    @classmethod
    def create(Class, random, state_info):
        raise NotImplementedError()

    def update(self, base_reward, is_done, state_info):
        """Returns modified reward and current task status"""
        raise NotImplementedError()

    def finished(self):
        return self.status != TaskStatus.RUNNING

    def __str__(self):
        return '{}: t={}, st={}'.format(
            type(self).__name__,
            self.n_steps,
            self.status.name
        )

    def as_info_dict(self):
        """A multi task environment should return it's current task in the form of info dict
        from next(action) and reset()"""
        return self._info_dict

class DummyTask(WarehouseTask):
    task_id = 0
    @classmethod
    def is_available(Class, state_info):
        return True

    @classmethod
    def create(Class, random, state_info):
        return Class()

    def update(self, base_reward, is_done, state_info):
        self.n_steps += 1
        self.status = TaskStatus.SUCCESS if is_done else TaskStatus.RUNNING
        return base_reward, self.status



class PickUp(WarehouseTask):
    """
    The goal is to pickup an item of specified type.
    """
    required_state_vars = {'room_id', 'item_count', 'item_id'}
    task_id = 1

    def __init__(self, target_item_id, room_id, duration=None, property_offset=1):
        info_dict = {'task_id':self.task_id,
                     'property':target_item_id+property_offset}
        super(PickUp, self).__init__(duration=duration, info_dict=info_dict)
        self.target_id = target_item_id
        self.room_id = room_id


    @classmethod
    def is_available(Class, state_info):
        room_id = state_info.room_id
        has_items = any(state_info.item_count)
        #hands are empty and there are items near the agent
        return (has_items > 0) and (state_info.item_id == Class.none_value)

    @classmethod
    def create(Class, random, state_info):
        room_id = state_info.room_id
        item_types = [i for i,n in enumerate(state_info.item_count) if n > 0]
        item = random.choice(item_types)
        return Class(target_item_id=item, room_id=room_id, property_offset=1) # offset of 1 accounts for the NoProperty entry.

    def update(self, base_reward, is_done, state_info):
        self.n_steps += 1
        item_id = state_info.item_id
        reward = base_reward
        if item_id == self.target_id:
            self.status = TaskStatus.SUCCESS
            reward += self.goal_reward
        elif self.n_steps >= self.duration: #if time is up:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        elif item_id != self.none_value: #if picks up wrong item:
            #reward += self.strong_penalty
            reward += self.fail_penalty # Lets punish it more!
            self.status = TaskStatus.FAIL
        elif state_info.room_id != self.room_id:
            reward += self.penalty
        return reward, self.status

    def __str__(self):
        return "Pick up item#{} in the room#{} [t={},{}]".format(
            self.target_id,
            self.room_id,
            self.n_steps,
            self.status.name
        )


class Drop(WarehouseTask):
    task_id = 2
    items_limit = 5
    required_state_vars = {'room_id', 'item_id', 'item_count'}

    def __init__(self, start_room_id, item_id, duration=None, property_offset=1):
        info_dict = {'task_id': self.task_id, 'property':item_id+property_offset}
        self.start_room_id = start_room_id
        super(Drop, self).__init__(duration=duration, info_dict=info_dict)


    @classmethod
    def is_available(Class, state_info):
        if state_info.item_id == Class.none_value:
            return False
        if state_info.room_id == Class.none_value:
            return False
        if sum(state_info.item_count) >= Class.items_limit:
            return False
        return True

    @classmethod
    def create(Class, random, state_info):
        item_id = state_info.item_id
        room_id = state_info.room_id
        return Class(start_room_id=room_id, item_id=item_id, property_offset=1) #offset of 1 accounts the NoProperty entry

    def update(self, base_reward, is_done, state_info):
        self.n_steps += 1
        reward = base_reward
        if state_info.room_id != self.start_room_id: #if agent left the room
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        elif state_info.item_id == self.none_value: # if item was dropped
            self.status = TaskStatus.SUCCESS
            reward += self.goal_reward
        elif self.n_steps >= self.duration:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        return reward, self.status

    def __str__(self):
        return "Drop carried item in the room#{} [t={},{}]".format(
            self.start_room_id,
            self.n_steps,
            self.status.name
        )


class Visit(WarehouseTask):
    """
    Visit specified room without picking up any item along the way
    """
    required_state_vars = {'room_id', 'rooms', 'item_id'}
    task_id = 3

    @classmethod
    def is_available(Class, state_info):
        if state_info.item_id != Class.none_value:
            return False

        curr_room = state_info.rooms.get(state_info.room_id, None)
        curr_texture = curr_room.texture if curr_room else None
        for r_id, room in state_info.rooms.items():
            if room.texture != curr_texture:
                return True
        return False

    @classmethod
    def create(Class, random, state_info):
        curr_room = state_info.rooms.get(state_info.room_id, None)
        curr_texture = curr_room.texture if curr_room else None
        rooms = [r for r_id, r in state_info.rooms.items()
                 if r.texture != curr_texture]
        target_room = random.choice(rooms)
        # All tasks has some meaningful property:
        # the movement tasks has room_texture_id(integers in the range [0,num_textures))
        # while manipulation tasks specify an item type (integers in the range [0,num_item_types)),
        # but later at the algorithm side we want to distinguish a texture with id=k with an item with type=k,
        # without knowing the nature of the property.
        # Therefore we just shift the texture ids by the n_items + one no_property to the left with property_offset
        # Yes, this is a cheap hack.
        n_items = len(state_info.item_count)+1
        return Class(target_room.id, target_room.texture,
                     property_offset=n_items)

    def __init__(self, target_id, texture, duration=None, property_offset=0):
        info = {'task_id':self.task_id, 'property': texture+property_offset}
        super(Visit, self).__init__(duration=duration, info_dict=info)
        self.target_id = target_id
        self.property = texture

    def update(self, base_reward, is_done, state_info):
        self.n_steps += 1
        reward = base_reward
        current_room = state_info.rooms.get(state_info.room_id,None)
        if state_info.item_id != self.none_value:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        #elif state_info.room_id == self.target_id
        elif current_room and current_room.texture == self.property:
            self.status = TaskStatus.SUCCESS
            reward += self.goal_reward
        elif self.n_steps >= self.duration:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        return reward, self.status

    def __str__(self):
        return "Visit room#{} [t={},{}]".format(
            self.target_id,
            self.n_steps,
            self.status.name
        )


class CarryItem(WarehouseTask):
    """
    Carry a picked up item to the specified room without dropping it
    """
    required_state_vars = {'room_id', 'rooms', 'item_id'}
    task_id = 4

    @classmethod
    def is_available(Class, state_info):
        if state_info.item_id == Class.none_value:
            return False

        curr_room = state_info.rooms.get(state_info.room_id, None)
        curr_texture = curr_room.texture if curr_room else None
        for r_id, room in state_info.rooms.items():
            if room.texture != curr_texture:
                return True
        return False

    @classmethod
    def create(Class, random, state_info):
        curr_room = state_info.rooms.get(state_info.room_id, None)
        curr_texture = curr_room.texture if curr_room else None

        rooms = [
            r for r in state_info.rooms.values()
            if r.texture != curr_texture
        ]
        target_room = random.choice(rooms)
        n_items = len(state_info.item_count) + 1 #see Visit.create comment
        return Class(target_room.id, target_room.texture,
                     state_info.item_id, property_offset=n_items)

    def __init__(self, target_room, texture, carried_item_id,
                 duration=None, property_offset=0):
        info = {'task_id': self.task_id, 'property':texture+property_offset}
        super(CarryItem, self).__init__(duration=duration, info_dict=info)
        self.target_id = target_room
        self.property = texture
        self.carried_item_id = carried_item_id

    def update(self, base_reward, is_done, state_info):
        self.n_steps += 1
        reward = base_reward
        curr_room = state_info.rooms.get(state_info.room_id,None)
        if state_info.item_id != self.carried_item_id:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        elif curr_room and curr_room.texture == self.property:
            self.status = TaskStatus.SUCCESS
            reward += self.goal_reward
        elif self.n_steps >= self.duration:
            self.status = TaskStatus.FAIL
            reward += self.fail_penalty
        return reward, self.status

    def __str__(self):
        return "Carry item#{} in the room#{} [t={},{}]".format(
            self.carried_item_id,
            self.target_id,
            self.n_steps,
            self.status.name
        )


class AbstractTaskManager(object):

    def next(self, **state_info):
        raise NotImplementedError('This method is not implemented yet!')

    def required_state_info(self):
        raise NotImplementedError('This method is not implemented yet!')


class TaskManager(AbstractTaskManager):
    """
    Creates a new task available in the current game state.
    The task will belong to one of the task types passed
    to the TaskManager instance.
    """

    def __init__(self, task_types, priorities=None):
        """

        :param task_types: A list of task classes to choose(a next task) from
        :param priorities: a type priority affects the probability that a next task would be an instance of this type.
        """
        self._task_types = np.asarray(task_types)
        if not priorities:
            num_tasks = len(task_types)
            self._priorities = np.full(len(task_types), 1./num_tasks)
        else:
            self._priorities = np.asarray(priorities)/sum(priorities)

    def next(self, state_info, rnd_state):
        is_available = [t.is_available(state_info) for t in self._task_types]
        priorities = self._priorities[is_available]
        task_types = self._task_types[is_available]
        print('available tasks:', [t.__name__ for t in task_types])
        selected_task_type = rnd_state.choice(task_types,
                                              p=priorities/sum(priorities))
        task = selected_task_type.create(rnd_state, state_info)
        print('selected task:', task, 'as_dict:', task.as_info_dict())
        return task

    def required_state_info(self):
        all_required_vars = set()
        all_vars = [task_cls.required_state_vars for task_cls in self._task_types]
        all_required_vars.update(*all_vars)
        return all_required_vars


class DummyManager(TaskManager):
    def __init__(self, *args, **kwargs):
        super(DummyManager, self).__init__([DummyTask], *args, **kwargs)


class AskHumanManager(TaskManager):
    """Asks(via console) a user to specify a type of the task that will be created"""
    def next(self, state_info, rnd_state):
        is_available = [t.is_available(state_info) for t in self._task_types]
        task_types = self._task_types[is_available]
        options = [t.__name__.lower() for t in task_types]
        answer = None
        while answer not in options:
            answer = input('Choose one of the following tasks: {}'.format(options))
            if answer.lower() not in options:
                print("Can't recognize the task. Please, try again.")
        id = options.index(answer)
        task = task_types[id].create(rnd_state, state_info)
        return task

