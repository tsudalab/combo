import numpy as np
import copy
import combo.misc
import pickle
from .results import history
from .. import utility
from ...variable import variable
from ..call_simulator import call_simulator
from ... import predictor
from ...gp import predictor as gp_predictor
from ...blm import predictor as blm_predictor
import combo.search.score
MAX_SEACH = int(20000)


class policy:
    def __init__(self, test_X, config=None):
        self.predictor = None
        self.training = variable()
        self.test = self._set_test(test_X)
        self.actions = np.arange(0, self.test.X.shape[0])
        self.history = history()
        self.config = self._set_config(config)

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

        self.new_data = variable(X, t, Z)
        self.history.write(t, action)
        self.training.add(X=X, t=t, Z=Z)

    def random_search(self, max_num_probes, num_search_each_probe=1,
                      simulator=None, is_disp=True):

        N = int(num_search_each_probe)

        if int(max_num_probes) * N > len(self.actions):
            raise ValueError('max_num_probes * num_search_each_probe must \
                be smaller than the length of candidates')

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            if is_disp and N > 1:
                utility.show_start_message_multi_search(self.history.num_runs)

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def bayes_search(self, training=None, max_num_probes=None,
                     num_search_each_probe=1,
                     predictor=None, is_disp=True,
                     simulator=None, score='TS', interval=0,
                     num_rand_basis=0):

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        self.training = self._set_training(training)

        if predictor is None:
            self.predictor = self._init_predictor(is_rand_expans)
        else:
            self.predictor = predictor

        N = int(num_search_each_probe)

        for n in range(max_num_probes):

            if utility.is_learning(n, interval):
                self.predictor.fit(self.training, num_rand_basis)
                self.test.Z = self.predictor.get_basis(self.test.X)
                self.training.Z = self.predictor.get_basis(self.training.X)
                self.predictor.prepare(self.training)
            else:
                try:
                    self.predictor.update(self.training, self.new_data)
                except:
                    self.predictor.prepare(self.training)

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search(self.history.num_runs,
                                                        score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self.get_actions(score, N, K, alpha)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def get_score(self, mode, predictor=None, training=None, alpha=1):
        self._set_training(training)
        self._set_predictor(predictor)
        actions = self.actions

        test = self.test.get_subset(actions)
        if mode == 'EI':
            f = combo.search.score.EI(predictor, training, test)
        elif mode == 'PI':
            f = combo.search.score.PI(predictor, training, test)
        elif mode == 'TS':
            f = combo.search.score.TS(predictor, training, test, alpha)
        else:
            raise NotImplementedError('mode must be EI, PI or TS.')
        return f

    def get_marginal_score(self, mode, chosed_actions, N, alpha):
        f = np.zeros((N, len(self.actions)))
        new_test = self.test.get_subset(chosed_actions)
        virtual_t \
            = self.predictor.get_predict_samples(self.training, new_test, N)

        for n in range(N):
            predictor = copy.deepcopy(self.predictor)
            train = copy.deepcopy(self.training)
            virtual_train = new_test
            virtual_train.t = virtual_t[n, :]

            if virtual_train.Z is None:
                train.add(virtual_train.X, virtual_train.t)
            else:
                train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            try:
                predictor.update(train, virtual_train)
            except:
                predictor.prepare(train)

            f[n, :] = self.get_score(mode, predictor, train)
        return f

    def get_actions(self, mode, N, K, alpha):
        f = self.get_score(mode, self.predictor, self.training, alpha)
        temp = np.argmax(f)
        action = self.actions[temp]
        self.actions = self.delete_actions(temp)

        chosed_actions = np.zeros(N, dtype=int)
        chosed_actions[0] = action

        for n in range(1, N):
            f = self.get_marginal_score(mode, chosed_actions[0:n], K, alpha)
            temp = np.argmax(np.mean(f, 0))
            chosed_actions[n] = self.actions[temp]
            self.actions = self.delete_actions(temp)

        return chosed_actions

    def get_random_action(self, N):
        random_index = np.random.permutation(range(self.actions.shape[0]))
        index = random_index[0:N]
        action = self.actions[index]
        self.actions = self.delete_actions(index)
        return action

    def load(self, file_history, file_training=None, file_predictor=None):
        self.history.load(file_history)

        if file_training is None:
            N = self.history.total_num_search
            X = self.test.X[self.history.chosed_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training = variable(X=X, t=t)
        else:
            self.training = variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, 'rb') as f:
                self.predictor = pickle.load(f)

    def export_predictor(self):
        return self.predictor

    def export_training(self):
        return self.training

    def export_history(self):
        return self.history

    def _set_predictor(self, predictor=None):
        if predictor is None:
            predictor = self.predictor
        return predictor

    def _init_predictor(self, is_rand_expans, predictor=None):
        self.predictor = self._set_predictor(predictor)
        if self.predictor is None:
            if is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)

        return self.predictor

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

    def _set_config(self, config=None):
        if config is None:
            config = combo.misc.set_config()
        return config
