import numpy as np


def show_search_results(history, N):
    n = history.total_num_search
    index = np.argmax(history.fx[0:n])

    if N == 1:
        print '%04d-th step: f(x) = %f (action=%d)' \
            % (n, history.fx[n-1], history.chosed_actions[n-1])
        print '   current best f(x) = %f (best action=%d) \n' \
            % (history.fx[index], history.chosed_actions[index])
    else:
        print 'current best f(x) = %f (best action = %d) ' \
            % (history.fx[index], history.chosed_actions[index])

        print 'list of simulation results'
        st = history.total_num_search - N
        en = history.total_num_search
        for n in xrange(st, en):
            print 'f(x)=%f (action = %d)' \
                % (history.fx[n], history.chosed_actions[n])
        print '\n'


def show_start_message_multi_search(N, score=None):
    if score == 'EI':
        print '%04d-th multiple probe search (EI) \n' % (N+1)
    elif score == 'PI':
        print '%04d-th multiple probe search (PI) \n' % (N+1)
    elif score == 'TS':
        print '%04d-th multiple probe search (TS) \n' % (N+1)
    else:
        print '%04d-th multiple probe search (random) \n' % (N+1)


def show_interactive_mode(simulator, history):
    if simulator is None and history.total_num_search == 0:
        print 'interactive mode stars ... \n '


def length_vector(t):
    N = len(t) if hasattr(t, '__len__') else 1
    return N


def is_learning(n, interval):
    if interval == 0:
        return True if n == 0 else False
    elif interval > 0:
        return True if np.mod(n, interval) == 0 else False
    else:
        return False
