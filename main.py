from Erdos_graph import plot_erdos_renyi_graph
from Small_world import generate_small_world_problem, create_small_world_edges
from The_growth_model import plot_growth_model
from Graph_Calculator import calculate_clustering_coefficient_from_edges, clustering_coefficient_exact, \
    calculate_clustering_coefficient
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def run_task_1():
    n = 100
    p = 0.3
    print("Running Erdos-Renyi random graph")
    print("%d nodes with connection probability %1.2f" % (n, p))
    plot_erdos_renyi_graph(n, p)


def run_task_2():
    n_const = 20  # Nodes
    c_const = 4  # Nearest neighbors should be even
    p_const = 0.2  # Rewiring probability
    print('Running Small world problem')
    print(
        '%d nodes connected to %d nearest neighbors with reconnection probability %1.2f' % (n_const, c_const, p_const))
    generate_small_world_problem(n_const, c_const, p_const)


def run_task_3():
    m = 2  # new number of connections at each time step
    n0 = m + 10  # Number of connected nodes t=0
    time_steps = 10  # Number of time steps
    print('Running The growth model')
    print('%d Number of connected nodes at t=0. %d New connections at each time step for %d time steps' %
          (n0, m, time_steps))
    plot_growth_model(m, n0, time_steps)


def task_4_evaluation(n_const, c_const, p_const):
    print(
        '%d nodes connected to %d nearest neighbors with reconnection probability %1.2f' % (n_const, c_const, p_const))
    edges, shortcut_edges = create_small_world_edges(n_const, c_const, p_const)
    exact_coefficient = clustering_coefficient_exact(c_const)
    calculated_cluster_coefficient = calculate_clustering_coefficient_from_edges(shortcut_edges)
    print('Exact clustering coefficient: %2.4f Calculated clustering coefficient: %2.4f'
          % (exact_coefficient, calculated_cluster_coefficient))


def run_task_4():
    n_const = 250  # Nodes
    c_const = 4  # Nearest neighbors should be even
    p_const = 0.0  # Rewiring probability
    print('Running Small world problem')
    print('----------------------------------------')
    task_4_evaluation(n_const, c_const, p_const)
    print('----------------------------------------')
    n_const = 500
    task_4_evaluation(n_const, c_const, p_const)
    print('----------------------------------------')
    n_const = 500
    c_const = 8
    task_4_evaluation(n_const, c_const, p_const)
    print('----------------------------------------')
    n_const = 500
    c_const = 4
    task_4_evaluation(n_const, c_const, p_const)
    print('-----------------------------------')
    graph = np.loadtxt('DataFiles/smallWorldExample.txt')
    coefficient = calculate_clustering_coefficient(graph)
    print('smallWorldExample.txt Cluster coefficient: %2.6f' % coefficient)
    graph = nx.Graph(graph)

    plt.subplot(1, 1, 1)
    nx.draw(graph)
    plt.title('smallWorldExample.txt Cluster coefficient: %2.6f' % coefficient)
    plt.show()


def main(task_nbr):
    if task_nbr == 1:
        run_task_1()
    elif task_nbr == 2:
        run_task_2()
    elif task_nbr == 3:
        run_task_3()
    elif task_nbr == 4:
        run_task_4()


if __name__ == '__main__':
    main(1)
