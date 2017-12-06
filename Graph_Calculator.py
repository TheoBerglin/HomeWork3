import numpy as np
from Graph_handling import edges_to_graph
from tqdm import tqdm
from scipy.sparse import csr_matrix


def calculate_clustering_coefficient_from_edges(edges):
    graph = edges_to_graph(edges)
    return calculate_clustering_coefficient(graph)


def calculate_clustering_coefficient(graph):
    # Triangles
    graph = csr_matrix(graph)
    graph_trip = graph.dot(graph.dot(graph))
    n_triangles = np.sum(graph_trip.diagonal())
    # Triples, 3 nodes connected through 2 edges
    graph_double = graph.dot(graph)
    graph_double.setdiag(0, 0)
    n_triples = np.sum(graph_double)
    # No times 3
    return n_triangles / n_triples


def clustering_coefficient_exact(c):
    return (3 * c - 2) / (4 * c - 1)


def get_path_length_matrix(graph):
    connections = graph.copy()  # Keeps track so that connections between all nodes have been found
    tmp_graph = graph.copy()  # used for matrix multiplication
    n = graph.shape[0]  # Size of the graph
    graph = csr_matrix(graph)  # No need to multiply elements that are zero
    end_run = n ** 2
    np.fill_diagonal(connections, 1)
    for l in range(2, n):  # Maximimum distance between nodes is n-1
        if np.count_nonzero(tmp_graph) == end_run:  # Path between all edges found
            return connections
        tmp_graph = graph.dot(tmp_graph)  # A^n
        new_connections = np.sign(connections) - np.sign(tmp_graph)  # If negative, new connection found
        if len(new_connections) > 0:
            connections[np.where(new_connections == -1)] = l  # Add new connection to the matrix of connections
        new_connections = None  # Reset for memory saving

    return connections


def calculate_path_length(connections_matrix):
    np.fill_diagonal(connections_matrix, 0)
    l = np.sum(connections_matrix)
    n = connections_matrix.shape[0]
    return l / (n * (n - 1))


def get_network_diameter(graph):
    connections = get_path_length_matrix(graph)
    return np.max(connections)


def get_path_length(graph):
    connections = get_path_length_matrix(graph)
    return calculate_path_length(connections)


def get_network_size(graph):
    connections = get_path_length_matrix(graph)
    return calculate_path_length(connections), np.max(connections)


def mean_graph_distribution(edges):
    n_nodes = len(np.unique(edges[:, 0]))
    connections = np.zeros(n_nodes)
    for node in edges:
        connections[int(node[1])] += 1
    return np.mean(connections)


def get_connections(edges):
    n_nodes = int(np.max(np.unique(edges)))+1
    connections = np.zeros(n_nodes)
    for node in edges:
        connections[int(node[1])] += 1
       # connections[int(node[0])] += 1
    return connections


def graph_distribution(edges):
    n_nodes = len(np.unique(edges[:, 0]))
    connections = np.zeros(n_nodes)
    for node in edges:
        connections[int(node[1])] += 1
    distributions = np.zeros(n_nodes)
    for c in connections:
        distributions[int(c)] += 1
    return distributions
