import numpy as np
import networkx as Nx
import matplotlib.pyplot as plt
from scipy.special import binom


def generate_positions(n, p):
    graph = np.zeros([n, n])
    for row in range(n):
        for col in range(row, n):
            if np.random.rand() < p:
                graph[row, col] = 1
                graph[col, row] = 1

    return graph


def analyze_graph(graph):
    x = np.zeros(graph.shape[0])
    for col in range(graph.shape[0]):
        x[int(np.sum(graph[:, col]))] += 1
    return x


def plot_erdos_renyi_graph(n, p):
    # Graph data
    graph = generate_positions(n, p)
    graph_dist = analyze_graph(graph)
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
