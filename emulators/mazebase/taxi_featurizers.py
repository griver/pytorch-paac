from mazebase.games import featurizers as mb_featurizers
from itertools import product
import numpy as np

class TaxiFeaturizer(mb_featurizers.BaseGridFeaturizer):
    OUT_OF_BOUND = 'OUT_OF_BOUND'

    def __init__(self, **kwargs):
        self.notify = kwargs.pop('notify', False)
        super(TaxiFeaturizer, self).__init__(**kwargs)

    def _featurize_side_info(self, game, id):
        return game._side_information()[0]

    def all_possible_features(self, game):
        fts = super(TaxiFeaturizer, self).all_possible_features(game)
        fts.append(self.OUT_OF_BOUND)
        return fts


class LocalViewFeaturizer(TaxiFeaturizer):
    '''
    When passed to a game, LocalViewFeaturizer returns environment observations
    in a matrix of width by height centered around an agent.
    kwargs:
        window_size - width and height of a window around an agent.
            To create a symmetrical window around an agent
            window_size values should be odd integers.
        notify - if True sets feature 'OUT_OF_BOUND' in the elements of
            the window showing locations beyond the game map borders.
    '''
    def __init__(self, **kwargs):
        size = kwargs.pop('window_size', (5,5))
        w,h = (size,size) if isinstance(size, int) else size
        if w%2 == 0: w += 1
        if h%2 == 0: h += 1
        self.window_size = (w,h)
        super(LocalViewFeaturizer, self).__init__(**kwargs)

    def _featurize_grid(self, game, item_id):
        ax,ay = game._items[item_id].location
        w, h = self.window_size
        features = [[[] for _ in range(h)] for _ in range(w)]
        min_map_x, min_map_y = ax - w//2, ay - h//2
        for x,y in product(range(w),range(h)):
            map_x = min_map_x + x
            map_y = min_map_y + y
            if not ((0 <= map_x < game.width) and (0 <= map_y < game.height)):
                if self.notify:
                    features[x][y].append(self.OUT_OF_BOUND)
                continue
            for item in game._map[map_x][map_y]:
                if item.visible:
                    features[x][y].extend(item.featurize())

        return features


class GlobalViewFeaturizer(TaxiFeaturizer):

    def _featurize_grid(self, game, id):
        max_w, max_h = game.get_max_bounds()
        w, h = game.width, game.height

        features = [[[] for y in range(max_w)]
                    for x in range(max_h)]

        for (x, y) in product(range(max_w), range(max_h)):
            if x < w and y < h:
                itemlst = game._map[x][y]
                for item in itemlst:
                    if not item.visible:
                        continue
                    features[x][y].extend(item.featurize())
            elif self.notify:
                features[x][y].append(self.OUT_OF_BOUND)

        return features


class FewHotEncoder(object):
    DEFAULT_FEATURES = [
        'Water', 'Corner', 'SingleTileMovable',
        'Agent', 'goal_id0', 'Passenger',
        'psg0_in_taxi', 'psg0', 'OUT_OF_BOUND',
        'Goal', 'Block'
    ]

    def __init__(self, selected_features=None):
        if selected_features is None:
            selected_features = FewHotEncoder.DEFAULT_FEATURES
        self.selected_features = selected_features
        self.feat2id = {sf: i for i, sf in enumerate(self.selected_features)}

    def encode(self, observation):
        xm, ym, zm = len(observation), len(observation[0]), len(self.feat2id)
        arr = np.zeros((xm, ym, zm), dtype=np.float32)
        for x, y in product(range(xm), range(ym)):
            for feat in observation[x][y]:
                arr[x][y][self.feat2id[feat]] = 1
        return arr