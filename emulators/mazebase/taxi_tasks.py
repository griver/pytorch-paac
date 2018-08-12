from enum import IntEnum

class TaskStatus(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    FAIL = 2

class TaxiTask(object):
    task_id = -1

    def __init__(self,
                 duration=None,
                 status=TaskStatus.RUNNING):
        super(TaxiTask, self).__init__()
        self.status = status
        self.n_step = 0
        self.duration = duration

    def update_status(self, *args, **kwargs):
        raise NotImplementedError()

    def finished(self):
        return self.status == TaskStatus.SUCCESS

    def failed(self):
        return self.status == TaskStatus.FAIL

    def as_info_dict(self):
        return {
            'task_id':self.task_id,
            'status':int(self.status)
        }


class DummyTask(TaxiTask):
    task_id = 0


class PickUp(TaxiTask):
    task_id = 1


class DropOff(TaxiTask):
    task_id = 2


class ReachPassanger(TaxiTask):
    task_id = 3


class ReachDestination(TaxiTask):
    task_id = 4
