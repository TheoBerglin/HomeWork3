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
