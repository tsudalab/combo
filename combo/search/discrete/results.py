import numpy as np
from .. import utility
MAX_SEARCH = int(30000)


class history:
    def __init__(self):
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros(MAX_SEARCH, dtype=float)
        self.chosed_actions = np.zeros(MAX_SEARCH, dtype=int)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

    def write(self, t, action):
        N = utility.length_vector(t)
        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en] = t
        self.chosed_actions[st:en] = action
        self.num_runs += 1
        self.total_num_search += N

    def export_sequence_best_fx(self):
        best_fx = np.zeros(self.num_runs)
        best_actions = np.zeros(self.num_runs)
        for n in xrange(self.num_runs):
            index = np.argmax(self.fx[0:self.terminal_num_run[n]])
            best_actions[n] = self.chosed_actions[index]
            best_fx[n] = self.fx[index]

        return best_fx, best_actions

    def export_all_sequence_best_fx(self):
        best_fx = np.zeros(self.total_num_search)
        best_actions = np.zeros(self.total_num_search)
        best_fx[0] = self.fx[0]
        best_actions[0] = self.chosed_actions[0]

        for n in xrange(1, self.total_num_search):
            if best_fx[n-1] < self.fx[n]:
                best_fx[n] = self.fx[n]
                best_actions[n] = self.chosed_actions[n]
            else:
                best_fx[n] = best_fx[n-1]
                best_actions[n] = best_actions[n-1]

        return best_fx, best_actions

    def save(self, filename):
        N = self.total_num_search
        M = self.num_runs
        np.savez_compressed(filename, num_runs=M, total_num_search=N,
                            fx=self.fx[0:N],
                            chosed_actions=self.chosed_actions[0:N],
                            terminal_num_run=self.terminal_num_run[0:M])

    def load(self, filename):
        data = np.load(filename)
        M = data['num_runs']
        N = data['total_num_search']
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data['fx']
        self.chosed_actions[0:N] = data['chosed_actions']
        self.terminal_num_run[0:M] = data['terminal_num_run']
