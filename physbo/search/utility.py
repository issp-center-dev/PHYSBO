import numpy as np


def show_search_results(history, N):
    n = history.total_num_search
    index = np.argmax(history.fx[0:n])

    if N == 1:
        print(
            "%04d-th step: f(x) = %f (action=%d)"
            % (n, history.fx[n - 1], history.chosen_actions[n - 1])
        )
        print(
            "   current best f(x) = %f (best action=%d) \n"
            % (history.fx[index], history.chosen_actions[index])
        )
    else:
        print(
            "current best f(x) = %f (best action = %d) "
            % (history.fx[index], history.chosen_actions[index])
        )

        print("list of simulation results")
        st = history.total_num_search - N
        en = history.total_num_search
        for n in range(st, en):
            print("f(x)=%f (action = %d)" % (history.fx[n], history.chosen_actions[n]))
        print("\n")


def show_search_results_mo(history, N, disp_pareto_set=False):
    n = history.total_num_search
    pset, step = history.pareto.export_front()

    def msg_pareto_set_updated(indent=False):
        prefix = "   " if indent else ""
        if history.pareto.front_updated:
            print(prefix + "Pareto set updated.")
            if disp_pareto_set:
                print(
                    prefix
                    + "current Pareto set = %s (steps = %s) \n"
                    % (str(pset), str(step + 1))
                )
            else:
                print(
                    prefix + "the number of Pareto frontiers = %s \n" % str(len(step))
                )

    if N == 1:
        print(
            "%04d-th step: f(x) = %s (action = %d)"
            % (n, str(history.fx[n - 1]), history.chosen_actions[n - 1])
        )

        msg_pareto_set_updated(indent=True)

    else:
        msg_pareto_set_updated()

        print("list of simulation results")
        st = history.total_num_search - N
        en = history.total_num_search
        for n in range(st, en):
            print(
                "f(x) = %s (action = %d)"
                % (str(history.fx[n]), history.chosen_actions[n])
            )
        print("\n")


def show_start_message_multi_search(N, score=None):
    if score is None:
        score = "random"
    print(f"{N+1:04}-th multiple probe search ({score})")


def show_interactive_mode(simulator, history):
    if simulator is None and history.total_num_search == 0:
        print("interactive mode starts ... \n ")


def length_vector(t):
    N = len(t) if hasattr(t, "__len__") else 1
    return N


def is_learning(n, interval):
    if interval == 0:
        return n == 0
    elif interval > 0:
        return np.mod(n, interval) == 0
    else:
        return False
