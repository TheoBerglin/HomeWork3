import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_empty_graph(nodes):
    return np.zeros([nodes, nodes])


def add_nearest_neighbours(graph, c):
    n = graph.shape[0]
    for node in range(n):
        for neighbor_index in range(1, int(c / 2) + 1):
            graph[node, (node + neighbor_index) % n] = 1
            graph[node, (node - neighbor_index) % n] = 1
    return graph


def add_shortcuts(graph, n_edges, p):
    n = graph.shape[0]
    graph_tmp = graph.copy()
    for i in range(n_edges):
        add_shortcut = np.random.rand()
        if add_shortcut < p:
            row = np.random.randint(0, n)
            col = np.random.randint(0, n)
            graph_tmp[row, col] = 1
    return graph_tmp


def generate_small_world_problem(n, c, p):
    graph = create_empty_graph(n)
    graph = add_nearest_neighbours(graph, c)
    n_edges = int(np.sum(graph)/2)
    shortcut_graph = add_shortcuts(graph, n_edges, p)
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
