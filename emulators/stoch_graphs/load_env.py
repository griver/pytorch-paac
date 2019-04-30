from os.path import join as join_path
import os
import numpy as np
import json
from .stoch_envs import Point
from networkx.drawing import nx_pydot


def load(resource_folder, env_name, create_env):
    topology, size, id = env_name.split('-')
    if topology == 'torus':
        env = load_torus(resource_folder, env_name, create_env)
    else:
        env = load_from_pydot(resource_folder, env_name, create_env)
    return env


def load_torus(resource_folder, game, create_env, ext='json'):
    filename = get_filename(resource_folder, game, 'json')

    probs = get_json_data(filename)
    get_probs = lambda n: probs[:n]

    _, size, _ = game.split('-')
    x,y = map(int, size.split('x'))
    env = build_torus(x, y, create_env, get_probs)
    return env


def get_filename(resource_folder, game, ext):
    topology, size, id = game.split('-')
    path = join_path(resource_folder, topology, size, id)
    return path + os.path.extsep + ext


def get_json_data(filename):
    """Reads data from json file"""
    data = None
    with open(filename, "r") as the_file:
        data = json.load(the_file)

    return data


def build_torus(x, y, create_env=None, get_probabilities=None):

    size = int(np.ceil(np.log2(x*y)))
    binary = '{0:0' + str(size) + 'b}'
    env = create_env(size, start_state_id=0)
    points = np.empty((x, y), dtype=Point)
    edges = []

    for i in range(x):
        for j in range(y):
            points[i][j] = Point(tuple(binary.format(i*y + j)))
            env.add_vertex(points[i][j])

    for i in range(x):
        for j in range(y):
            e1e2 = env.add_both_edges(points[i][j], points[i][(j + 1) % y])  # right
            edges.append(e1e2)
            e1e2 = env.add_both_edges(points[i][j], points[(i + 1) % x][j])  # bottom
            edges.append(e1e2)

    if get_probabilities is None:
        return env

    probs = get_probabilities(len(edges))  # число пар ребер в два раза больше числа вершин

    for i in range(len(edges)):
        env.create_united_group(edges[i], probs[i])
    return env


def load_from_pydot(resource_folder, game, create_env):
    filename = get_filename(resource_folder, game, 'dot')
    nx_graph = read_nx_graph_from_dot(filename)
    env = nx_graph_to_env(nx_graph, create_env)
    return env


def read_nx_graph_from_dot(filename):
    """
    Read networkx graph from filename
    :param filename: name of the dot file with graph info
    :return: networkx graph
    """
    graph = nx_pydot.read_dot(filename)
    #for e in graph.edges_iter(data=True):
    for e, e_data in graph.edges.items():
        #e is a tuple(node_1, node2, key==0)
        #e_data is a dict with properties of e
        e_data['is_available'] = e_data['is_available'] == 'True'
        e_data['prob'] = float(e_data['prob'].strip('"'))
    return graph


def nx_graph_to_env(nx_graph, create_env=None):
    size_of_state_vector = int(np.ceil(np.log2(nx_graph.number_of_nodes())))
    binary = '{0:0' + str(size_of_state_vector) + 'b}'

    env = create_env(size_of_state_vector, start_state_id=0)
    print("environment type is:", type(env))
    points = np.empty(nx_graph.number_of_nodes(),dtype=Point)

    for i in range(nx_graph.number_of_nodes()):
        points[i] = Point(tuple(binary.format(i)))
        env.add_vertex(points[i])

    #for node1, node2, edge_data in nx_graph.edges_iter(data=True):
    for e, e_data in nx_graph.edges.items():
        node1, node2, _ = e
        e1e2 = env.add_both_edges(points[int(node1)], points[int(node2)])
        env.create_united_group(e1e2, e_data['prob'])


    assert len(env._stoch_groups) == nx_graph.number_of_edges(), \
        "Ошибка, число ребер в nx_graph'e должно равнятся числу пар ребер в среде"
    return env
