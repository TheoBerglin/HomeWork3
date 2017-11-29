from Erdos_graph import plot_erdos_renyi_graph


def run_task_1():
    n = 400
    p = 0.1
    print("Running Erdos-Renyi random graph")
    print("%d nodes with connection probability %1.2f" % (n, p))
    plot_erdos_renyi_graph(n, p)


def main(task_nbr):
    if task_nbr == 1:
        run_task_1()

if __name__ == '__main__':
    main(1)
