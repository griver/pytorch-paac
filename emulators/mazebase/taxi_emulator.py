from .mazebase_emulator import *

from .taxi_game_objects import OldTaxi, FewHotEncoder, FewHotEncoderPlus
from .taxi_game import Taxi, FixedTaxi
from .taxi_plus_game import TaxiPlus
import mazebase.items as maze_items
import os.path as path
from utils.allsubclasses import find_subclass
import glob
import cv2
import imageio

TAXI_CLASSES = frozenset([
    OldTaxi,
    Taxi,
    FixedTaxi,
    TaxiPlus
])

TAXI_GAMES = {camel_to_snake_case(cls.__name__):cls for cls in TAXI_CLASSES}


class MapViewer(object):
    ITEM_2_NAME = {
        'Water':'water',
        'Block':'block',
        'Goal':'target_32',
        'Passenger':'passenger_32',
        'Cargo':'cargo_32',
        'RestrainedMultiTaskTaxiAgent':'car_32',
        'TaxiAgent':'car_32',
    }

    def __init__(self, icon_folder, game, icon_size=32):
        self.icon_size = icon_size

        icons = glob.glob(path.join(icon_folder, '*.jpg'))
        icons.extend(glob.glob(path.join(icon_folder, '*.png')))
        icons.sort()

        icon_imgs = [cv2.imread(i) for i in icons]
        #only 32x32 images are allowed:
        predicate = lambda img:img.shape[0] == img.shape[1] == self.icon_size
        get_name = lambda n:path.splitext(path.basename(n))[0]
        self.name2icon = {
            get_name(icon):img
            for icon, img in zip(icons, icon_imgs)
            if predicate(img)
        }
        self.game = game

    def start_episode(self):
        self.images = []

    def save_episode(self, filename, folder='./pretrained/gif/'):
        assert len(self.images) > 0, 'I thought you wanted to save something..'
        imageio.mimsave(folder + filename, self.images)

    def show_map(self, ignore_frame=False):
        self.img_width = self.game.width * self.icon_size
        self.img_height = self.game.height * self.icon_size

        self.img = np.ones((self.img_height, self.img_width, 3), np.uint8) * 255

        for y in reversed(range(self.game.height)):
            for x in range(self.game.width):
                item_list = sorted(
                    filter(lambda x:x.visible, self.game._map[x][y]),
                    key=lambda x:x.PRIO
                )
                for item in item_list:
                    if self._needs_drawing(item):
                        y_hat = self.game.height - y - 1  # mazebase draws y-axis upwards!
                        self._draw_item(x, y_hat, item)

        im_title = "{}[{}x{}]".format(
            type(self.game).__name__,
            self.game.width,
            self.game.height
        )
        cv2.imshow(im_title, self.img)
        if not ignore_frame:
            self.images.append(self.img)

        return self.img

    def show_frequency(self, loc2freqs, title, alpha_lim=(0.05,1.), freq_color=(0,0, 255)):

        self.img_width = self.game.width * self.icon_size
        self.img_height = self.game.height * self.icon_size
        alpha_min, alpha_max = alpha_lim
        alpha_range = alpha_max - alpha_min

        self.img = np.ones((self.img_height, self.img_width, 3), np.uint8) * 255
        color_tile = np.ones((self.icon_size, self.icon_size, 3))*np.array(freq_color)

        for y in reversed(range(self.game.height)):
            y_hat = self.game.height - y - 1  # mazebase draws y-axis upwards!
            for x in range(self.game.width):
                item_list = sorted(
                    filter(lambda x:x.visible, self.game._map[x][y]),
                    key=lambda x:x.PRIO
                )
                for item in item_list:
                    if self._needs_drawing(item):
                        self._draw_item(x, y_hat, item)

                freq = loc2freqs.get((x,y), 0.)
                if freq > 0.:
                    freq = freq*alpha_range + alpha_min # squeeze in the right range
                    self._transparent_tile_add(x,y_hat, freq, color_tile)

        spec_objs = [self.game.target, self.game.passenger, self.game.agent]
        if hasattr(self.game, 'cargo'):
            spec_objs.append(self.game.cargo)

        for obj in spec_objs:
            x,y = obj.location
            y_hat = self.game.height - y - 1
            self._draw_item(x,y_hat, obj)

        im_title = "Location frequency for {}".format(title)
        cv2.imshow(im_title, self.img)

        return self.img

    def move_obj(self, obj_name, new_loc):
        """
        If, for example, agent ends up in the same location as the passenger,
        then one of them will cover the other on the image.
        The function is a fast kludge to show all desirable objects.
        We simply specify move one of the objects to a new location, e.g.
        move the agent to it's starting point when display visitation frequency heatmap!
        """
        if hasattr(self.game, obj_name):
            obj_id = getattr(self.game, obj_name).id
            self.game._move_item(obj_id, new_loc)
        else: # if, for some reason there is no such object create it!
            cls = find_subclass(maze_items.MazeItem, obj_name)
            if cls is None:
                raise ValueError("Can't find a mazeitem with name {}".format(obj_name))
            obj = cls(location=new_loc)
            setattr(self.game, obj_name, obj)
            self.game._add_item(obj)

    def _needs_drawing(self, item):
        """
        if an item has a special character for ascii art or a background color
        then it needs to be drawn
        :return( bool ) True if it needs to be drawn, False otherwise!
        """
        config = item._get_display_symbol()
        for v in config[1:]:
            if v is not None:
                return True

        s = config[0]
        if s is None or s.isspace():
            return False
        return True

    def _draw_item(self, x, y, item):
        name = self.ITEM_2_NAME.get(type(item).__name__, None)
        if name:
            self._draw_tile(x, y, self.name2icon[name])

    def _draw_tile(self, x, y, icon):
        mask = np.sum(icon, 2, keepdims=True) > 0
        self._transparent_tile_add(x, y, mask, icon)
        #z = self.icon_size
        #background = self.img[y * z: (y + 1) * z, x * z: (x + 1) * z]
        #result = background * (1. - mask) + icon * mask
        #self.img[y * z: (y + 1) * z, x * z: (x + 1) * z] = result  #icon.copy()

    def _transparent_tile_add(self, x, y, add_mask, addition):
        z = self.icon_size
        old_tile = self.img[y * z: (y + 1) * z, x * z: (x + 1) * z]
        new_tile =  old_tile*(1. - add_mask) + addition*add_mask
        self.img[y * z: (y + 1) * z, x * z: (x + 1) * z] = new_tile


class TaxiEmulator(MazebaseEmulator):
    """
    Adapts games from mazebase to the BaseEnvironment interface.
    """
    @staticmethod
    def available_games():
        return TAXI_GAMES

    def __init__(self, emulator_id, game, full_view=False, verbose=0,
                 view_size=DEFAULT_LOCAL_VIEW_SIZE, map_size=DEFAULT_MAP_SIZE,
                 random_seed=17, finish_action=False, fail_reward=0.,
                 single_task_episodes=False, max_episode_steps=300,
                 preliminary_env=False, tasks=None, visualize=False,
                 icon_folder=None,
                 **unknown):
        if verbose >= 2:
            logging.debug('Emulator#{} received unknown main_args: {}'.format(emulator_id, unknown))
        self.emulator_id = emulator_id
        available_games = self.available_games()
        assert game in available_games, '{0}: There is no such game in the mazebase framework'.format(game)
        game_cls = available_games[game]

        if full_view:
            featurizer = GlobalViewFeaturizer(notify=True)
        else:
            featurizer = LocalViewFeaturizer(window_size=view_size, notify=True)
        self._encoder = FewHotEncoderPlus() if game == 'taxi_plus' else FewHotEncoder()

        game_seed = (self.emulator_id * random_seed) % (2**32)
        self.game = game_cls(map_size=map_size,
                             featurizer=featurizer,
                             max_episode_steps=max_episode_steps,
                             random_seed=game_seed,
                             finish_action=finish_action,
                             fail_reward=fail_reward,
                             single_task_episodes=single_task_episodes,
                             preliminary_env=preliminary_env,
                             tasks=tasks)


        state, _, _, _ = self._observe() #masebase resets games during __init__
        self.observation_shape = state.shape
        self.legal_actions = self.game.actions()
        assert 'pass' in self.legal_actions, 'There should be noop action among the available actions!'
        self.noop = 'pass'
        self.id = emulator_id

        if visualize and icon_folder:
            self.map_viewer = MapViewer(icon_folder, self.game)
        else:
            self.map_viewer = None

        if verbose > 2:
            logging.debug('Intializing mazebase.{0} emulator_id={1}'.format(game, self.id))
        # Mazebase generates random samples
        # within itself in different modules across the package; therefore,
        # we can't fix a random generator without rewriting mazebase.
        # self.rand_gen = random.Random(main_args.random_seed)


    def reset(self):
        #There is no activity in the masebase games aside from agent's activity.
        #Therefore, performing a random start changes nothing in the games
        if self.map_viewer:
            self.map_viewer.start_episode()

        self.game.reset()
        state, _, _, info = self._observe()
        #state = self._merge_observation_with_task_info(state, info)
        return state, info

    def _observe(self):
        # returns s, r, is_done, info
        game_data = self.game.observe()
        state, info = game_data['observation'] #info looks like this: [str(task_name), int(task_id)]
        state = self._encoder.encode(state)
        state = state.transpose((2,0,1)) #images go in the [C,H,W] shape for pytorch
        return state, game_data['reward'], self.game.is_over(), info

    def next(self, action):
        '''
        Performs action.
        Returns the next state, reward, and termination signal
        '''
        action = self.legal_actions[action]
        self.game.act(action) # no need for action repetition here
        state, reward, is_done, info = self._observe()
        return state, reward, is_done, info

    def display(self, ignore_frame=False):
        state, reward, is_done, info = self._observe()
        if self.game.episode_steps <= 0:
            print('=============== EM#{} ============'.format(self.emulator_id))
        if self.map_viewer:
            self.map_viewer.show_map(ignore_frame=ignore_frame)
            print('r_t={:.2f} task_t+1={} | info_t+1={}'.format(reward, self.game.task(), info))
        else:
            print('r_t={:.2f} task_t+1={} | info_t+1={} | state_t+1:'.format(
                reward, self.game.task(), info
            ))
            self.game.display()

    def from_keyboard(self, allowed_keys=None):

        while True:
            if self.map_viewer:
                key = cv2.waitKey(0)
                key = chr(key)
            else:
                key = input('Input your action:\n')

            if (allowed_keys is None) or (key in allowed_keys):
                break
            else:
                print('{} is not allowed. Allowed keys are: {}'.format(key, allowed_keys))
        return key



    def set_map_size(self, min_x, max_x, min_y, max_y):
        return self.game.set_map_size(min_x, max_x, min_y, max_y)

    def update_map_size(self, *deltas):
        return self.game.update_map_size(*deltas)

    def get_tasks_history(self):
        return self.game.get_tasks_history()