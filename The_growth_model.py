import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Graph_help import edges_to_graph


def initialize_growth_edges(n0, m):
    graph = list()
    for node in range(n0):
        for neighbor_index in range(1, int(m / 2) + 1):
            graph.append((node, (node + neighbor_index) % n0))
            graph.append((node, (node - neighbor_index) % n0))

    return np.asarray(graph)


def run_growth_model(edges, n0, m, time_steps):
    t = 0
    while t < time_steps:
        nodes = edges.shape[0]
        connections = np.random.randint(0, nodes, m)

        new_connections = np.zeros([len(connections), 2])
        for i in range(len(connections)):
            new_connections[i, :] = [int(n0 + t), edges[int(connections[i]), 0]]
        edges = np.append(edges, new_connections, axis=0)
        t += 1
    return edges


def growth_model(m=2, n0=12, time_steps=10):
    edges = initialize_growth_edges(n0, m)
    return run_growth_model(edges, n0, m, time_steps)


def plot_growth_model(m=2, n0=12, time_steps=10):
    edges = growth_model(m, n0, time_steps)
    graph = edges_to_graph(edges)
    graph = nx.Graph(graph)
    nx.draw(graph)
    plt.show()


def main():
    plot_growth_model()

if __name__ == '__main__':
    main()
