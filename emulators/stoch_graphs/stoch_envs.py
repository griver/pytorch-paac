import numpy as np
from functools import reduce
import numpy.random as rnd
from . import graph_algs as algs



class Vertex(object):
    def __init__(self, name="none"):
        self.__id = -1
        self._incoming = []
        self._outcoming = []
        self.name = name


    def get_id(self):
        return self.__id

    def get_outcoming(self):
        return tuple(self._outcoming)

    def get_incoming(self):
        return tuple(self._incoming)

    def _set_id(self, id):  # only for graph class internal usage
        self.__id = id

    def _add_outcoming(self, edge):  # only for graph class internal usage
        if self._outcoming.count(edge) == 0:
            self._outcoming.append(edge)

    def _add_incoming(self, edge):  # only for graph class internal usage
        if self._incoming.count(edge) == 0:
            self._incoming.append(edge)

    def _remove_outcoming(self, edge):
        self._outcoming.remove(edge)

    def _remove_incoming(self, edge):
        self._incoming.remove(edge)

    def __str__(self):
        return str(self.get_id())


class Point(Vertex):
    def __init__(self, coordinates):
        Vertex.__init__(self, coordinates.__str__())
        self._coord = coordinates
        self.name = str(coordinates)

    def coords(self):
        return self._coord

    def get_dimension(self):
        return len(self._coord)

    def __repr__(self):
        return self.name


class Edge(object):
    def __init__(self, src, dst, weight):  # возможно стоит поменять weight на data
        self._src = src
        self._dst = dst
        self._weight = weight

        src._add_outcoming(self)
        dst._add_incoming(self)

    def get_src(self):
        return self._src

    def get_dst(self):
        return self._dst

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def remove(self):
        self._src._remove_outcoming(self)
        self._dst._remove_incoming(self)

    def __str__(self):
        result = "Src:" + str(self._src.get_id()) + " Dst:" + str(self._dst.get_id()) + " Weight:" + str(self._weight)

    def is_available(self):
        return True


class Graph(object) :

    def __init__(self, name) :
        self._vertex_list = []
        self._edge_list = []
        self._name = name

    def add_vertex(self, vertex):
        if self.is_contains_vertex(vertex) :
            return -1
        vertex._set_id(len(self._vertex_list))
        self._vertex_list.append(vertex)
        return vertex.get_id()

    def get_vertices_number(self):
        return len(self._vertex_list)

    def remove_vertex(self, vertex):
        if self.is_contains_vertex(vertex):
            self._vertex_list.pop(vertex)
            return True
        return False

    def remove_vertex_by_id(self, index):
        if self.__check_vertex_id(index):
            self._vertex_list.pop(index)
            return True
        return False

    def is_contains_vertex(self, item):
        if not self.__check_vertex_id(item.get_id()): return False
        return self._vertex_list[item.get_id()] == item

    def get_vertex(self, id):
        return self._vertex_list[id]

    def vertices(self):
        return tuple(self._vertex_list)

    def add_edge(self, src, dst, weight=1):
        if self.is_contains_vertex(src) and self.is_contains_vertex(dst):
            return Edge(src, dst, weight)

    def add_both_edges(self, src, dst, weight=1):
        e1 = self.add_edge(src, dst, weight)
        e2 = self.add_edge(dst, src, weight)
        return e1, e2

    def remove_edge(self, edge):
        if self.is_contains_vertex(edge.get_src()) and self.is_contains_vertex(edge.get_dst()):
            edge.remove()
            return True
        return False

    def __check_vertex_id(self, id):
        if 0 <= id < self.get_vertices_number():
            return True
        return False

    def get_name(self):
        return self._name

    def __str__(self):
        result = "digraph " + self._name + " {\n"
        for v in self._vertex_list:
            result += "   v"+ str(v.get_id()) + " [label=\""+v.name+"\"];\n"

        for v in self._vertex_list:
            for e  in v.get_outcoming():
                #if(e.is_available()): #1
                result += "   v" + str(v.get_id()) + " -> v" + str(e.get_dst().get_id())
                result += " [label=\""+ str(e.weight()) + "\"];\n"

        result += "}"
        return result

    def write_to_file(self, filename = None):
        if filename is None: filename = self._name + ".dot"
        f = open(filename, 'w')
        f.write(self.__str__())
        f.close()


class Environment(Graph):
    def __init__(self, dimension, name="environment", start_state_id=0,):
        Graph.__init__(self, name)
        self._dim = dimension # the dimension of a space
        self._curr_state = None
        self._start_id = start_state_id

    def get_dimension(self):
        return self._dim

    def distance(self, src, dst):
        #if(src.get_dimension() != dst.get_dimension()):
        dist = 0
        for i in range(0, src.get_dimension()):
            dist += abs(src.coords()[i] - dst.coords()[i])

        return dist**(1.0/2.0)  # Euclidean norm

    def reset(self):
        if len(self.vertices()) > self._start_id:
            self.set_current_state(self.get_vertex(self._start_id))

    def distance_from_current(self, state):  # get distance from current environment state
        return self.distance(self.get_current_state(), state)

    def get_state_by_coords(self, coords):
        if len(coords) != self.get_dimension():
            raise ValueError("invalide number of coordinates")
        return next((p for p in self._vertex_list if (p.coords() == coords)), None)

    def get_current_state(self):
        #if self._curr_state is None:
        #    self.reset()
        return self._curr_state

    def set_current_state(self, state):
        if self.is_contains_vertex(state):
            self._curr_state = state
            return True
        return False

    def has_path_to(self, target_state):
        return target_state is algs.bfs(self, self.get_current_state(), target_state)

    def update_state(self, index):
        curr = self.get_current_state()
        tmp_st = None

        #ищем вершину с заданными координатами среди смежных вершин
        #for e in curr.get_outcoming():
        #    if e.get_dst().coords() == coords:
        if index < len(curr.get_outcoming()):
            e = curr.get_outcoming()[index]
            if e.is_available():
                tmp_st = e.get_dst()

        if tmp_st is None:
            #if self._msg:
                #print("Нельзя перейти из " + curr.name + " в " + str(coords))
            #    self._msg = False
            return

        #print("Перешли из " + curr.name + " в " + tmp_st.name)
        self.set_current_state(tmp_st)
        #self._msg = True


class StochasticTransitsGroup(object):
    def __init__(self):
        self.edges = []
        self.mins = []
        self.maxs = []
        self.value = None

    def next_value(self):
        self.value = rnd.random()

    def add_edge(self, edge, min_val, max_val):
        self.edges.append(edge)
        self.mins.append(min_val)
        self.maxs.append(max_val)
        edge.set_stochastic_group(self)

    def remove_edge(self, edge):
        i = self.edges.index(edge)
        self.edges.remove(edge)
        self.mins.remove(self.mins[i])
        self.maxs.remove(self.maxs[i])

    def recalc(self):
        self.next_value()
        for i in range(0, len(self.edges)):
            self.edges[i].set_availability(self.mins[i] <= self.value < self.maxs[i])

    def clear(self):
        for e in self.edges:
            e.set_availability(True)
            e.set_stochastic_group(None)


class StochasticTransit(Edge):
    _available = True
    _stochastic_group = None

    def set_stochastic_group(self, transit_group):
        self._stochastic_group = transit_group

    def set_availability(self, val):
        self._available = val

    def is_available(self):
        return self._available

    def get_stochastic_group(self):
        return self._stochastic_group


class StochasticEnvironment(Environment):


    def __init__(self, dimension, name="Environment", start_state_id=0):
        Environment.__init__(self, dimension, name, start_state_id)
        self._stoch_groups = []

    def create_competitive_group(self, edges, probs=tuple()):
        prob_sum = reduce(lambda x, y: x+y, probs, 0)
        edges_len = len(edges)
        prob_len = len(probs)

        if prob_len == edges_len:
            if prob_sum > 1.0:
                raise ValueError("sum of the probabilities must be <= 1.0")
        elif prob_len < edges_len:
            if prob_sum >= 1.0:
                raise ValueError("sum of the probabilities must be less than 1.0")

        if reduce(lambda x, y: x * (y.get_stochastic_group() is None), edges, True) is False:
            print("at least one of edges already has a group!")

        group = StochasticTransitsGroup()
        curr = 0.0
        for i in range(0, edges_len):
            if prob_len > i:
                max_val = curr + probs[i]
            else:
                max_val = curr + (1.0 - curr)/(edges_len - i)
            group.add_edge(edges[i], curr, max_val)
            curr = max_val
        self._stoch_groups.append(group)
        group.recalc()

    def create_united_group(self, edges, probability):
        probability = min(1.0, probability)

        if reduce(lambda x, y: x * (y.get_stochastic_group() is None), edges, True) is False:
            print("at least one of edges already has a group!")

        group = StochasticTransitsGroup()

        for i in range(len(edges)):
            group.add_edge(edges[i], 0.0, probability)

        self._stoch_groups.append(group)
        group.recalc()

    def create_arbitrary_group(self, *edges2prob):
        '''
        Receives an arbitrary number of 2-tuples of the form: (l, p) where
           l - a list of edges
           p - probability of edges availability
        The method produces a stochastic group of edges where all edges from one tuple
        compose a united group. And all united groups constitutes a competitive meta-group(group of groups).
        For example, if the method is called with following arguments:
          create_arbitrary_group( ([e1,e3], 0.3), ([e2,e4], 0.7))
        then the edges e1 and e3(or e2 and e4) will be always available or not available together.
        And if edges from first group is available then edges from the second groups is not!
        '''
        total_prob = sum(p for e,p in edges2prob)
        if total_prob < 0. or total_prob > 1.:
            raise ValueError('Sum of probabilities must be in the unit interval!')

        group = StochasticTransitsGroup()
        min_val = 0.
        for edges, p in edges2prob:
            max_val = min_val + p
            for e in edges:
                group.add_edge(e, min_val, max_val)

            min_val = max_val
        self._stoch_groups.append(group)
        group.recalc()

    def add_edge(self, src, dst, weight=1):
        if self.is_contains_vertex(src) and self.is_contains_vertex(dst):
            return StochasticTransit(src, dst, weight)

    def reset(self):
        Environment.reset(self)
        for g in self._stoch_groups:
            g.recalc()


    def remove_stochasticity(self):
        for g in self._stoch_groups:
            g.clear()
        self._stoch_groups = []


class PeriodicEnvironment(StochasticEnvironment):
    """
    Стохастическая среда, где успешность действий разыгрывается с определенной частотой
    """

    def __init__(self, period, dimension, name="PeriodicEnvironment", start_state_id = 0):
        super(PeriodicEnvironment, self).__init__(dimension, name, start_state_id)
        self.period = max(1, period)
        self._counter = 0
        self.retry_update_limit = 100
        self.max_safe_period = 5

    def update_state(self, index):

        super(PeriodicEnvironment, self).update_state(index)

        self._counter += 1
        if self._counter >= self.period:
            self._counter = 0
            self._update_all_transits()


    def _update_all_transits(self):
        for g in self._stoch_groups:
            g.recalc()

    def reset(self):
        """ Сейчас переводит среду в начальное состояние и обновляет доступность всех переходов """
        super(PeriodicEnvironment, self).reset()
        self._counter = 0

    def ensure_path_reset(self, target):
        self.reset()
        if (self.period <= self.max_safe_period) or self.has_path_to(target):
            return True

        return self._ensure_path_update(target)

    def _ensure_path_update(self, target):
        for _ in range(self.retry_update_limit):
            self._update_all_transits()
            if self.has_path_to(target):
                return True

        return False

    def ensure_path_update_state(self, index, target):
        self.update_state(index)
        if (self.period <= self.max_safe_period) or self.has_path_to(target):
            return True

        return self._ensure_path_update(target)

    # def has_path_to(self, target):
    #     if self.period > np.sqrt(self.get_vertices_number()):
    #         return super(PeriodicEnvironment, self).has_path_to(target)
    #     return True  # среда обновляется с переодичностью так, что рано или поздно путь откоется.
