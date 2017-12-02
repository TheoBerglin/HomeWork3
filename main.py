from Erdos_graph import plot_erdos_renyi_graph
from Small_world import generate_small_world_problem
from The_growth_model import plot_growth_model


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


def main(task_nbr):
    if task_nbr == 1:
        run_task_1()
    elif task_nbr == 2:
        run_task_2()
    elif task_nbr == 3:
        run_task_3()


if __name__ == '__main__':
    main(3)
