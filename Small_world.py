import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Graph_help import edges_to_graph


def create_nearest_neighbour_edges(n, c):
    edges = list()
    for node in range(n):
        for neighbor_index in range(1, int(c / 2) + 1):
            edges.append((node, (node + neighbor_index) % n))
            edges.append((node, (node - neighbor_index) % n))
    return np.asarray(edges)


def add_shortcuts(edges, n_nodes, p):
    new_edges = list()
    for i in range(edges.shape[0]):
        add_shortcut = np.random.rand()
        if add_shortcut < p:
            row = np.random.randint(0, n_nodes)
            col = np.random.randint(0, n_nodes)
            new_edges.append((row, col))

    return np.append(edges, np.asarray(new_edges), axis=0)


def generate_small_world_problem(n, c, p):
    edges = create_nearest_neighbour_edges(n, c)
    shortcut_edges = add_shortcuts(edges, n, p)
    graph = edges_to_graph(edges)
    shortcut_graph = edges_to_graph(shortcut_edges)

    graph = nx.Graph(graph)
    shortcut_graph = nx.Graph(shortcut_graph)

    # Plotting
    plt.subplot(1, 2, 1)
    circular_pos = nx.layout.circular_layout(graph)
    nx.draw(graph, circular_pos)
    plt.title('No shortcuts: n=%d, c=%d' % (n, c))
    plt.subplot(1, 2, 2)
    circular_pos_shortcut = nx.layout.circular_layout(shortcut_graph)
    nx.draw(shortcut_graph, circular_pos_shortcut)
    plt.title('Shortcuts added: p=%1.2f' % p)

    plt.show()


if __name__ == '__main__':
    n_const = 40  # Nodes
    c_const = 4  # Nearest neighbors should be even
    p_const = 0.2  # Rewiring probability
    generate_small_world_problem(n_const, c_const, p_const)
