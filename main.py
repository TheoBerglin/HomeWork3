from Erdos_graph import plot_erdos_renyi_graph
from Small_world import generate_small_world_problem


def run_task_1():
    n = 40
    p = 0.1
    print("Running Erdos-Renyi random graph")
    print("%d nodes with connection probability %1.2f" % (n, p))
    plot_erdos_renyi_graph(n, p)


def run_task_2():
    n_const = 20  # Nodes
    c_const = 4  # Nearest neighbors should be even
    p_const = 0.2  # Rewiring probability
    print('Running Small world problem')
    print('%d nodes connected to %d nearest neighbors with reconnection probability %1.2f' % (n_const, c_const, p_const))
    generate_small_world_problem(n_const, c_const, p_const)


def main(task_nbr):
    if task_nbr == 1:
        run_task_1()
    elif task_nbr == 2:
        run_task_2()

if __name__ == '__main__':
    main(1)
