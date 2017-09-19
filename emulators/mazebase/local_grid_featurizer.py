from mazebase.games import featurizers
from itertools import product

class LocalGridFeaturizer(featurizers.BaseGridFeaturizer):
    '''
    When passed to a game, LocalGridFeaturizer returns environment observations
    in a matrix of width by height centered around an agent.
    kwargs:
        window_size - width and height of a window around an agent.
            To create a symmetrical window around an agent
            window_size values should be odd integers.
        notify - if True sets feature 'OUT_OF_BOUND' in the elements of
            the window showing locations beyond the game map borders.
    '''
    OUT_OF_BOUND = 'OUT_OF_BOUND'

    def __init__(self, **kwargs):
        size = kwargs.pop('window_size', (5,5))
        w,h = (size,size) if isinstance(size, int) else size
        if w%2 == 0: w += 1
        if h%2 == 0: h += 1
        self.window_size = (w,h)
        self.notify = kwargs.pop('notify', False)
        super(LocalGridFeaturizer, self).__init__(**kwargs)

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

    def all_possible_features(self, game):
        fts = super(LocalGridFeaturizer, self).all_possible_features(game)
        fts.append(self.OUT_OF_BOUND)
        return sorted(fts)