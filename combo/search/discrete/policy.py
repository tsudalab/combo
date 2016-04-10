import numpy as np
import copy
from results import history
import util
from ...variable import variable
from ..call_simulator import call_simulator
#import combo.search.score as score
MAX_SEACH = int(20000)


class policy:
    def __init__(self, test_X, config):
        self.predictor = None
        self.training = variable()   # training data
        self.test = self._set_test(test_X)  # all candidates
        self.actions = np.arange(0, self.test.X.shape[0])
        self.history = history()

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)

    def delete_actions(self, index, actions=None):
        actions = self._set_unchosed_actions(actions)
        return np.delete(actions, index)

    def write(self, action, t, X=None):
        if X is None:
            X = self.test.X[action, :]
            Z = self.test.Z[action, :] if self.test.Z is not None else None
        else:
            Z = self.predictor.get_basis(X) \
                if self.predictor is not None else None

        self.history.write(t, action)
        self.training.add(X=X, t=t, Z=Z)

    def random_search(self, max_num_probes, num_search_each_probe=1,
                      simulator=None, is_disp=True):

        N = int(num_search_each_probe)

        if int(max_num_probes) * N > len(self.actions):
            raise ValueError('max_num_probes * num_search_each_probe must \
                be smaller than the length of candidates')

        if is_disp:
            util.show_interactive_mode(simulator, self.history)

        for n in xrange(0, max_num_probes):

            if is_disp and N > 1:
                util.show_start_message_multi_search(self.history.num_runs)

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                util.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def bayes_search(self, training, max_num_probes=None,
                     num_search_each_probe=1,
                     predictor=None, is_disp=True,
                     simulator=None, score='TS', interval=0,
                     num_rand_basis=0):

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        self.training = self._set_training(training)
        self.predictor = self._init_predictor(is_rand_expans)

        N = int(num_search_each_probe)

        for n in xrange(max_num_probes):
            if util.is_learning(n, interval):
                predictor.fit(training, self.config, num_rand_basis)
                self.test.Z = predictor.get_basis(self.test.X)
                self.train.Z = predictor.get_basis(self.train.X)
                self.prepare(new_data)

            action = self.get_action(score, N, alpha=self.config.search.alpha)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                util.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def prepare(self, new_data=None):
        if new_data is None:
            self.predictor.prepare(self.training)
            self.training.Z = self.predictor.get_basis(self.training.X)
            self.test.Z = self.predictor.get_basis(self.test.X)
        else:
            try:
                self.predictor.update(self.training, new_data)
                self.training.add(X=new_data.X, t=new_data.t, Z=new_data.Z)
            except:
                self.training.add(X=new_data.X, t=new_data.t, Z=new_data.Z)
                self.predictor.prepare(self.training)



    def get_random_action(self, N):
        random_index = np.random.permutation(xrange(self.actions.shape[0]))
        index = random_index[0:N]
        action = self.actions[index]
        self.actions = self.delete_actions(index)
        return action

    def load(self, filename, training=None, predictor=None):
        self.history.load(filename)

        if training is None:
            N = self.history.total_num_search
            X = self.test.X[self.history.chosed_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training = training = variable(X=X, t=t)
        else:
            self.training = training

        self.predictor = predictor

    def export_predictor(self):
        return self.predictor

    def export_training(self):
        return self.training

    def _set_predictor(self, predictor=None):
        if predictor is None:
            predictor = self.predictor
        return predictor

    def _set_training(self, training=None):
        if training is None:
            training = self.training
        return training

    def _set_unchosed_actions(self, actions=None):
        if actions is None:
            actions = self.actions
        return actions

    def _set_test(self, test_X):
        if isinstance(test_X, np.ndarray):
            test = variable(X=test_X)
        elif isinstance(test_X, variable):
            test = test_X
        else:
            raise TypeError('The type of test_X must \
                             take ndarray or combo.variable')
        return test
