from collections import deque


def bfs(graph, source, target):
    queue = deque([source])
    visited = {source}

    while queue:
        vertex = queue.popleft()
        if vertex is target: return target

        verts = [e.get_dst() for e in vertex.get_outcoming() if e.is_available() and e.get_dst() not in visited]
        visited.update(verts)
        queue.extend(verts)

    return None


def eulerian_cycle(graph, source):
    if filter(lambda x: len(x.get_outcoming()) != len(x.get_incoming()), graph.vertices()):
        return None

    stack = [source]
    index = {v: -len(v.get_outcoming()) for v in graph.vertices()}
    result = []

    while stack:
        v = stack[-1]
        i = index[v]
        if not index[v]:
            result.append(stack.pop())
        else:
            stack.append(v.get_outcoming()[i].get_dst())
            index[v] = i + 1

    return result


def dfs_explore(graph, source, visited, get_neighbors, pre_visit=lambda x: None, post_visit=lambda x: None):
    visited.add(source)
    pre_visit(source)
    for v in get_neighbors(source):
        if v not in visited:
            dfs_explore(graph, v, visited, get_neighbors, pre_visit, post_visit)
    post_visit(source)


def topological_sort(graph, get_neighbors):
    vertices = graph.vertices()
    visited = set()
    top_sort = deque()

    pre_visit = lambda x: None

    def post_visit(vertex):
        top_sort.appendleft(vertex)

    for v in vertices:
        if v not in visited:
            dfs_explore(graph, v, visited, get_neighbors, pre_visit, post_visit)

    return top_sort


def strong_connected_components(graph):
    def get_incoming(vertex):
        return [e.get_src() for e in vertex.get_incoming() if e.is_available()]

    def get_outcoming(vertex):
        return [e.get_dst() for e in vertex.get_outcoming() if e.is_available()]

    top_sort = topological_sort(graph, get_incoming)

    visited = set()
    scc = deque()

    def pre_visit(vertex):
        scc[0].add(vertex)

    for v in top_sort:
        if v not in visited:
            scc.appendleft(set())
            dfs_explore(graph, v, visited, get_outcoming, pre_visit)

    return scc

