import numpy as np
from Graph_help import edges_to_graph


def calculate_clustering_coefficient_from_edges(edges):
    graph = edges_to_graph(edges)
    return calculate_clustering_coefficient(graph)


def calculate_clustering_coefficient(graph):
    # Triangles
    graph_trip = np.dot(np.dot(graph, graph), graph)
    n_triangles = np.sum(graph_trip.diagonal())
    # Triples, 3 nodes connected through 2 edges
    graph_double = np.dot(graph, graph)
    np.fill_diagonal(graph_double, 0)
    n_triples = np.sum(graph_double)
    # No times 3
    return n_triangles/n_triples


def clustering_coefficient_exact(c):
    return (3*c-2)/(4*c-1)

def get_path_length_matrix(graph):
    connections = graph.copy()
    tmp_graph = graph.copy()
    n = graph.shape[0]
    for l in range(2, n):
        if np.min(connections) > 0:
            return connections
        tmp_graph = np.dot(tmp_graph, graph)
        for col in range(n):
            for row in range(n):
                if connections[row, col] == 0:
                    if tmp_graph[row, col] > 0:
                        connections[row, col] = l
    return connections

def calculate_path_length(connections_matrix):
    np.fill_diagonal(connections_matrix, 0)
    l = np.sum(connections_matrix)
    n = connections_matrix.shape[0]
    return l/(n*(n-1))

def get_path_length(graph):
    connections = get_path_length_matrix(graph)
    return calculate_path_length(connections)

