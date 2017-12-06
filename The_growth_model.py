import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Graph_handling import edges_to_graph
from Graph_Calculator import get_connections
from tqdm import tqdm

def initialize_growth_edges(n0, m):
    #return np.random.randint(0, n0, [n0*m, 2])
    graph = list()
    for node in range(n0):
        for neighbor_index in range(1, int(m / 2) + 1):
            graph.append((node, (node + neighbor_index) % n0))
            graph.append((node, (node - neighbor_index) % n0))

    return np.asarray(graph)


def run_growth_model(edges, n0, m, time_steps):
    t = 0
    pbar = tqdm(total=time_steps)
    new_way = True
    while t < time_steps:

        if new_way:
            nodes = np.asarray(edges[:, 1].copy())
            connections = list()
            for i in range(m):
                new_rand = np.random.randint(0,nodes.shape[0])
                new_node = nodes[new_rand]
                while new_node in connections:
                    new_rand = np.random.randint(0, nodes.shape[0])
                    new_node = nodes[new_rand]
                connections.append(new_rand)
                #np.delete(nodes, np.where(nodes == new_node))
        else:
            n_nodes = edges.shape[0]
            connections = np.random.randint(0, n_nodes, m)

        new_connections = np.zeros([2*len(connections), 2])

        for i in range(len(connections)):
            # Undirected
            new_connections[2*i, :] = [int(n0 + t), edges[int(connections[i]), 1]]
            new_connections[2*i+1, :] = [edges[int(connections[i]), 1], int(n0 + t)]
        edges = np.append(edges, new_connections, axis=0)
        pbar.update(1)
        t += 1
    pbar.close()
    return edges


def growth_model(m=2, n0=12, time_steps=10):
    edges = initialize_growth_edges(n0, m)
    return run_growth_model(edges, n0, m, time_steps)


def plot_growth_model(m=2, n0=12, time_steps=10):
    edges = growth_model(m, n0, time_steps)
    n_connections = get_connections(edges)
    n = len(n_connections)
    n_connections = np.sort(n_connections)[::-1]
    y_data = [i/n for i in range(1, n+1)]
    graph = edges_to_graph(edges)
    #plt.loglog(n_connections, y_data)

    graph = nx.Graph(graph)
    plt.figure(1)
    plt.subplot(1,1,1)
    nx.draw(graph)
    plt.title('The preferential growth model: m = %d, $\\eta_0$ = %d, time steps: %d' % (m, n0, time_steps))
    plt.show()


def main():
    plot_growth_model()


def plot_degree_distribution():
    m_const = 5
    n0 = 5
    time_steps = 20000
    edges = growth_model(m_const,n0,time_steps)
    n_connections = get_connections(edges)
    print(np.max(np.unique(edges[:, 0])))
    # n_connections = np.sort(n_connections)[::-1]
    n_connections = np.sort(n_connections)[::-1]
    n_connections = np.trim_zeros(n_connections)
    #n_connections = n_connections[np.where(n_connections < 10000)]
    print(n_connections)
    n = len(n_connections)
    y_data = [i / n for i in range(1, n + 1)]
    #x_data = 1 / n_connections
    #x_data = n_connections/np.max(n_connections)
    x_data = n_connections
    # graph = edges_to_graph(edges)
    plt.loglog(x_data, y_data, label='Simulation')

    k = np.linspace(np.max(n_connections), np.min(n_connections), 100)
    print(k)
    #k = np.sort(k)[::-1]
    #y_power = [2*(m_const**2)*k(i)**(-2) for i in k]
    y_power = [2*(m_const/k_v)**2 for k_v in k]
    y_power = np.sort(y_power)[::-1]
    #k = np.sort(k)[::-1]
    #plt.plot()
    #x_d = 1/k
    #x_d = np.sort(1/k)
    new_number_of_nodes = int(np.max(n_connections))
    start = int(np.min(n_connections))
    print(new_number_of_nodes, start)
    theoretical_values = [2 * np.power(m_const, 2) * (1 / np.power(k, 2)) for k in range(start, new_number_of_nodes)]
    plt.loglog(range(start, new_number_of_nodes), theoretical_values/np.max(theoretical_values), label='Theoretical prediction')
    plt.title('Power law: m = %d, n0 = %d, time steps = %d' % (m_const, n0, time_steps))
    plt.xlabel('Degree')
    plt.ylabel('cCDF')
    plt.legend(loc='best')
    #plt.loglog(1/k, y_power)

    #plt.loglog(n_connections/np.max(n_connections), y_data)
    plt.show()


def theoretical_degree_prediction(m, k):
    gamma = 3
    return 2*m**2*k**(-gamma+1)

if __name__ == '__main__':
    #main()
    plot_degree_distribution()
