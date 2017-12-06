import numpy as np
import networkx as nx
from Graph_Calculator import *
from Graph_handling import edges_to_graph
import matplotlib.pyplot as plt

def open_network(network):
    return np.loadtxt(network)

def single_network_analyzis(network_edges, network_graph):
    n = network_graph.shape[0]
    print('Number of nodes n: %d' % n)
    m = network_edges.shape[0]
    print('Number of edges m: %d' % m)
    l, d = get_network_size(network_graph)
    print('Path length l: %.2f' % l)
    print('Diameter d: %.2f' % d)
    cluster_coefficient = calculate_clustering_coefficient(network_graph)
    print('Clustering coefficient C(1): %.3f' % cluster_coefficient)
    z = mean_graph_distribution(network_edges)
    print('Mean connection distribution z: %.2f' % z)
    return n, m, l, d, cluster_coefficient, z


def print_network_analyzis(network_edges, network_graph, network_name):
    #n, m, l, d, cluster_coefficient, z = single_network_analyzis(network_edges, network_graph)
    print("----------------------------------")
    print("Network name: %s" % network_name)
    n, m, l, d, cluster_coefficient, z = single_network_analyzis(network_edges, network_graph)


def plot_network_distributions(edges_1, edges_2,edges_3):
    print('hej')
    graph_dist_1 = graph_distribution(edges_1)
    graph_dist_1 = np.trim_zeros(graph_dist_1)
    x_1 = [i for i in range(1,len(graph_dist_1)+1)]
    plt.figure(1)

    plt.plot(x_1, graph_dist_1 / np.sum(graph_dist_1), linestyle="-")
    plt.title("Network 1")
    plt.xlabel('Degree')
    plt.ylabel('Proportion')
    print('hej')
    graph_dist_2 = graph_distribution(edges_2)
    graph_dist_2 = np.trim_zeros(graph_dist_2)
    x_2 = [i for i in range(1,len(graph_dist_2)+1)]
    plt.figure(2)
    plt.plot(x_2, graph_dist_2 / np.sum(graph_dist_2), linestyle="-")
    plt.title("Network 2")
    plt.xlabel('Degree')
    plt.ylabel('Proportion')
    print('hej')
    graph_dist_3 = graph_distribution(edges_3)
    graph_dist_3 = np.trim_zeros(graph_dist_3)
    x_3 = [i for i in range(1, len(graph_dist_3) + 1)]
    plt.figure(3)
    plt.plot(x_3, graph_dist_3 / np.sum(graph_dist_3), linestyle="-")
    plt.title("Network 3")
    plt.xlabel('Degree')
    plt.ylabel('Proportion')
    plt.show()
    print('hej')


def analyze_networks(network_1, network_2, network_3):
    network_1_edges = open_network(network_1)-1
    network_2_edges = open_network(network_2)-1
    network_3_edges = open_network(network_3)-1
    # Network graphs
    network_1_graph = edges_to_graph(network_1_edges)
    network_2_graph = edges_to_graph(network_2_edges)
    network_3_graph = edges_to_graph(network_3_edges)
    # Analysis

    print('Starting to perform network analysis')
    print_network_analyzis(network_1_edges, network_1_graph, 'Network 1')
    print_network_analyzis(network_2_edges, network_2_graph, 'Network 2')
    print_network_analyzis(network_3_edges, network_3_graph, 'Network 3')

    plot_network_distributions(network_1_edges, network_2_edges, network_3_edges)


