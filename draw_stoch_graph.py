import glob, os
import utils
import numpy as np


def get_filenames(topology, periods, graphs):
    filenames = dict()
    for p in periods:
        filenames[p] = {}
        for g in graphs:
            filenames[p][g] = os.path.join(
                'pretrained', 'stoch_graphs',
                'period_{}'.format(p),
                'a2c-{}-{}'.format(topology,g),
                'checkpoints/summaries.pkl4'
            )

    return filenames


def get_summaries(topology):
    periods = [1, 2, 5, 10, 20, 50, 100, 500]
    graphs = list(range(1, 11))
    filenames = get_filenames(topology, periods, graphs)

    def prepare_dict():
        return  {p:{g:[] for g in graphs} for p in periods}

    train_steps = prepare_dict()
    train_episodes = prepare_dict()
    mean_length = prepare_dict()
    std_length = prepare_dict()

    for p in filenames.keys():
        for g, filename in filenames[p].items():
            data = utils.load_summary(filename)
            for step, stats in data:
                # example of stats value:
                # {'quality': -217.7, 'mean_r': -1.6467, 'mean_steps': 217.7,
                # 'std_steps': 96.606, 'num_episodes': 0}
                train_steps[p][g].append(step)
                train_episodes[p][g].append(stats['num_episodes'])
                std_length[p][g].append(stats['std_steps'])
                mean_length[p][g].append(stats['mean_steps'])

    return train_steps, train_episodes, mean_length, std_length


def get_mean(data):

    def check_list_of_lists(ll):
        lengths = [len(l) for l in ll]
        assert min(lengths) == max(lengths), "lists should have the same length! But have {}".format(lengths)

    result = {}

    for period, d in data.items():
        values = [l for graph, l in d.items()]
        check_list_of_lists(values)
        values = np.array(values).mean(axis=0)
        result[period] = values
       # print('period={}, shape: {}'.format(period, values.shape))

    return result


if __name__ == '__main__':
    tr_steps, tr_eps, mean_length, std_length = get_summaries('torus-10x10')
    print('mean_length:')
    get_mean(mean_length)

    print('std_length:')
    get_mean(std_length)

    print('tr_eps:')
    get_mean(tr_eps)

    print('tr_steps:')
    get_mean(tr_steps)