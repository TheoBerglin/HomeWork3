import numpy as np


def edges_to_graph(edges):
    n_nodes = int(np.max(np.unique(edges)))+1
    graph = create_empty_graph(n_nodes)
    for edge in edges:
        graph[int(edge[0]), int(edge[1])] = 1
    return graph


def create_empty_graph(nodes):
    return np.zeros([nodes, nodes])
