from .taxi_game import Taxi, Relation
import itertools as it
import mazebase.items as maze_items
from mazebase import games
from mazebase.utils.mazeutils import  choice, MazeException
from mazebase.utils import creationutils

from .taxi_game_objects import Passenger, Cargo, TaxiAgent, rnd
from .taxi_tasks import TaskStatus
from . import taxi_tasks

class TaxiPlus(Taxi):
    """
    Version of TaxiMultiTaskGame where the
    Goal object can appear only in one of the four corners
    and the new cargo object is added to the map.
    """

    def __init__(self, *args, **kwargs):
        super(TaxiPlus, self).__init__(*args, **kwargs)
        features = super(TaxiPlus, self).all_possible_features()
        features.extend(Cargo.all_features())
        features.sort()

    def _reset(self):
        self.map_size = self.future_map_size
        #it is easier to first place goal object and then "sprinkle" water and
        super(games.WithWaterAndBlocksMixin, self)._reset()

        loc_target = self._choose_target_location()
        self.target = maze_items.Goal(location=loc_target)
        self._add_item(self.target)
        #place random walls and lakes at the map.
        creationutils.sprinkle(self, [(maze_items.Block, self.blockpct),
                                      (maze_items.Water, self.waterpct)])

        #we request more locations than actually needed to implicitly free up more space
        #in case the target is locked in the corner of the map:
        locations = self._get_placement_locs(self.target, 10)
        loc_agent, loc_passenger, loc_cargo = locations[:3]
        self.current_task = None

        init_state = self._choose_reset_config()
        # check relationship between locations of the passenger, taxi and target
        if init_state.taxi_destination == Relation.NEAR:
            loc_agent = loc_target

        if init_state.passenger_taxi != Relation.FAR:
            loc_passenger = loc_agent

        self.agent = self.agent_cls(location=loc_agent)
        self._add_agent(self.agent, "TaxiAgent")

        self.passenger = Passenger(location=loc_passenger)
        self._add_item(self.passenger)

        if init_state.passenger_taxi == Relation.INSIDE:
            self.agent.actions['pickup']()
            assert self.passenger.is_pickedup, "Can't put a passenger into a taxi for init_state={}".format(init_state)

        self.cargo = Cargo(location=loc_cargo)
        self._add_item(self.cargo)

        self.episode_steps = -1 # see the self._step() comment

        self._info = {}
        self._tasks_history = []

    def _respawn_objects_if_needed(self, finished_task):
        """
        Respawn passenger and cargo if needed
        """
        if finished_task.status != TaskStatus.RUNNING:
            respawn_list = []
            #respawn passenger if he has reached target location:
            if (self.target.location == self.passenger.location) \
            and (self.passenger.is_pickedup is False):
                respawn_list.append(self.passenger)
            #respawn cargo if agent has finished a task at the same location as target:
            if isinstance(finished_task, (taxi_tasks.ConveyPassenger, taxi_tasks.DropOffPassenger)):
                if self.cargo.is_pickedup:
                    self.agent.forced_dropoff()
                respawn_list.append(self.cargo)

            if respawn_list:
                #get 3 random(except locations of the agent and target)
                # locations for new placements:
                locs = self._get_placement_locs(self.agent, len(respawn_list) + 1)
                locs = [l for l in locs if l != self.target.location]
                for i, map_obj in enumerate(respawn_list):
                    self._move_item(map_obj.id, locs[i])

    def _choose_target_location(self):
        """
        Randomly returns one of the four map corners
        """
        y_vals = (0, self.height-1)
        x_vals = (0, self.width-1)
        return (choice(x_vals), choice(y_vals))