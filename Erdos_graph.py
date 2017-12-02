import numpy as np
import networkx as Nx
import matplotlib.pyplot as plt
from scipy.special import binom
from Graph_help import edges_to_graph


def generate_erdos_edges(n, p):
    return np.asarray([(i, j) for i in range(n) for j in range(n) if np.random.rand() < p])


def graph_distribution(edges):
    n_nodes = len(np.unique(edges[:, 0]))
    connections = np.zeros(n_nodes)
    for node in edges:
        connections[node[1]] += 1
    distributions = np.zeros(n_nodes)
    for c in connections:
        distributions[int(c)] += 1
    return distributions


def plot_erdos_renyi_graph(n, p):
    # Graph data
    edges = generate_erdos_edges(n, p)
    graph = edges_to_graph(edges)
    graph_dist = graph_distribution(edges)
    # Plot data
    g2 = Nx.Graph(graph)
    x = [i for i in range(n)]
    y_data = [binom(n - 1, k) * p ** k * (1 - p) ** (n - 1 - k) for k in range(n)]
    # Plotting
    plt.subplot(1, 2, 1)
    Nx.draw(g2)
    plt.title("Erdos-Renyi random graph, n=%d, p=%1.2f" % (n, p))
    plt.subplot(1, 2, 2)
    plt.title("Graph distribution")
    plt.plot(x, graph_dist / np.sum(graph_dist), linestyle=":", label='Graph dist')
    plt.plot(x, y_data, label="Theoretical")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    n_const = 300
    p_const = 0.2

    plot_erdos_renyi_graph(n_const, p_const)
