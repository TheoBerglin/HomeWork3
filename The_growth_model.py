import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def initialize_growth_graph(n0, m):
    graph = list()

    for node in range(n0):
        for neighbor_index in range(1, int(m / 2) + 1):
            graph.append((node, (node + neighbor_index) % n0))
            graph.append((node, (node - neighbor_index) % n0))

    return np.asarray(graph)


def edges_to_graph(edges):
    n_nodes = len(np.unique(edges[:, 0]))
    graph = np.zeros([n_nodes, n_nodes])
    for edge in edges:
        graph[int(edge[0]), int(edge[1])] = 1
    return graph


def run_growth_model(graph, n0, m, time_steps):
    t = 0

    while t < time_steps:
        nodes = graph.shape[0]
        connections = np.random.randint(0, nodes, m)

        new_connections = np.zeros([len(connections), 2])
        for i in range(len(connections)):
            new_connections[i, :] = [int(n0 + t), graph[int(connections[i]), 0]]
        graph = np.append(graph, new_connections, axis=0)
        t += 1
    return graph


def main():
    m = 2
    n0 = m + 10
    graph = initialize_growth_graph(n0, m)
    graph = run_growth_model(graph, n0, m, 10)
    graph = edges_to_graph(graph)

    graph = nx.Graph(graph)
    nx.draw(graph)
    plt.show()


if __name__ == '__main__':
    main()
