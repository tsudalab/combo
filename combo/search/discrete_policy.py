import numpy as np
import score
import os
from ..variable import variable
from ..gp import predictor as gp_predictor
from ..blm import predictor as blm_predictor


class result:
    def __init__(self, max_length_search=int(10000)):
        self.num_probe = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros(max_length_search, dtype=float)
        self.best_fx = np.zeros(max_length_search, dtype=float)
        self.best_action = np.zeros(max_length_search, dtype=int)
        self.multi_probe_best_fx = np.zeros(max_length_search, dtype=float)
        self.chosed_actions = np.zeros(max_length_search, dtype=int)

    def write(self, t, action, N):
        st = self.total_num_search
        en = st + N

        for n in xrange(st, en):
            self.fx[n] = t if N == 1 else t[n-st]

            if n == 0:
                temp_t = t if N == 1 else t[n-st]
                temp_action = action if N == 1 else action[n-st]
                self.best_fx[n] = temp_t
                self.best_action[n] = temp_action
            else:
                temp_t = t if N == 1 else t[n-st]
                temp_action = action if N == 1 else action[n-st]

                if self.best_fx[n-1] < temp_t:
                    self.best_fx[n] = temp_t
                    self.best_action[n] = temp_action
                else:
                    self.best_fx[n] = self.best_fx[n-1]
                    self.best_action[n] = self.best_action[n-1]

            self.chosed_actions[n] = action if N == 1 else action[n-st]

        max_t = t if N == 1 else np.max(t)

        k = self.num_probe
        if k == 0:
            self.multi_probe_best_fx[k] = max_t
        else:
            temp = self.multi_probe_best_fx[k-1]
            if temp < max_t:
                self.multi_probe_best_fx[k] = max_t
            else:
                self.multi_probe_best_fx[k] = temp

        self.total_num_search = en
        self.num_probe += 1

    def save(self, filename):
        N = self.total_num_search
        M = self.num_probe
        np.savez_compressed(
            filename, num_probe=M, total_num_search=N,
            fx=self.fx[0:N],
            best_fx=self.best_fx[0:N],
            best_action=self.best_action[0:N],
            multi_probe_best_fx=self.multi_probe_best_fx[0:M],
            chosed_actions=self.chosed_actions[0:N])

    def load(self, filename):
        data = np.load(filename)
        self.num_probe = data['num_probe']
        M = self.num_probe = data['num_probe']
        self.total_num_search = data['total_num_search']
        N = self.total_num_search
        self.fx[0:N] = data['fx']
        self.best_fx[0:N] = data['best_fx']
        self.best_action[0:N] = data['best_action']
        self.multi_probe_best_fx[0:M] = data['multi_probe_best_fx']
        self.chosed_actions[0:N] = data['chosed_actions']


class policy:
    def __init__(self, test_X, config):
        if isinstance(test_X, np.ndarray):
            self.test = variable(X=test_X)
        elif isinstance(test_X, variable):
            self.test = test_X
        else:
            raise TypeError('test_X must take ndarray or combo.variable')

        self.training = variable()
        self.unchosed_actions = np.arange(0, self.test.X.shape[0])
        self.config = config
        self.res = result()

    def load(self, filename, training=None, predictor=None):
        N = self.res.total_num_search
        M = self.res.num_probe
        self.res.load(filename)
        self.unchosed_actions = np.arrange(0, self.test.X.shape[0])
        self.unchosed_actions = np.delete(self.unchosed_actions,
                                          self.res.chosed_actions[0:M])
        if training is None:
            self.training.add(new_x=self.X[self.res.chosed_actions[0:M], :],
                              new_t=self.fx[0:N])
        else:
            self.training = training

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)

    def delete_unchosed_actions(self, n, unchosed_actions=None):
        if unchosed_actions is None:
            unchosed_actions = self.unchosed_actions

        unchosed_actions = np.delete(unchosed_actions, n)
        return unchosed_actions

    def get_score(self, chosed_actions=None, predictor=None):
        if predictor is None:
            predictor = self.predictor

        if chosed_actions is None:
            chosed_actions = self.res.chosed_actions

        if self.config.search.score == 'EI':
            f = score.EI(predictor, self.training, self.test)
        elif self.config.search.score == 'PI':
            f = score.PI(predictor, self.training, self.test)
        elif self.config.search.score == 'TS':
            f = score.TS(predictor, self.training, self.test,
                         alpha=self.config.search.alpha)
        else:
            raise NotImplementedError

        return f

    def call_simulator(self, simu, action):
        output = simu(action)
        if hasattr(output, '__len__') and len(output) == 2:
            t = output[0]
            x = output[1]
        else:
            t = output
            x = None  # self.test.X[action, :]
        return t, x

    def write(self, action, new_t, new_x=None, new_z=None):
        if hasattr(new_t, '__len__'):
            N = len(new_t)
        else:
            N = 1

        if new_x is None:
            new_x = self.test.X[action, :]
        self.res.write(new_t, action, N)
        self.training.add(new_x, new_t, new_z)

    def random_action(self, N):
        random_index \
            = np.random.permutation(xrange(self.unchosed_actions.shape[0]))

        index = random_index[0:N]
        action = self.unchosed_actions[index]
        self.unchosed_actions = self.delete_unchosed_actions(index)
        return action

    def get_learn_timinig(self, max_num_probe, learn_timing=None):
        if self.config.learning.is_hyparams_learning:
            if learn_timing is None:
                if self.config.learning.interval == 0:
                    learn_timing = np.array([0])
                else:
                    learn_timing = np.arange(0, max_num_probe,
                                             self.config.learning.interval)
        else:
            learn_timing = np.array([-1])
        return learn_timing

    def bayes_search(self, max_num_probe=1, num_search_each_probe=1,
                     simu=None, predictor=None, learn_timing=None,
                     traininig=None):

        if training is not None:
            self.training = training

        if predictor is not None:
            self.predictor = predictor

        N = int(num_search_each_probe)

        if int(max_num_probe) * N > self.test.X.shape[0]:
            raise ValueError('max_num_probe * num_search_each_probe must \
                              be smaller than the length of candidates')

        if self.config.search.is_disp and simu is None \
           and self.res.total_num_search == 0:
            print 'Interaction Mode Start'

        learn_timing = self.get_learn_timinig(learn_timing)
        num_do_learning = 0

        if self.is_learning(0, learn_timing, num_do_learning):
            self.learning()
            num_do_learning += 1

        for n in xrange(max_num_search):

            if self.config.search.is_disp and N > 1:
                if self.config.search.score == 'EI':
                    print '%04d-th multiple probe search (EI) \n' % (n+1)
                elif self.config.search.score == 'PI':
                    print '%04d-th multiple probe search (PI) \n' % (n+1)
                elif self.config.search.score == 'TS':
                    print '%04d-th multiple probe search (TS) \n' % (n+1)
                else:
                    raise NotImplementedError

            ''' generate actions '''
            action = self.get_action(N)

            if simu is None:
                return action

            new_t, new_x = self.call_simulator(simu, action)
            self.write(action, new_t, new_x)

            if self.config.search.is_disp:
                self.show(N)

            if self.is_learning():
                self.learning()
                num_do_learning += 1
            else:
                self.predictor = self.compute_stats()

    def get_action(self, N):
        if N == 1:
            score = self.get_score()
            index = np.argmax(score)
            action = self.unchosed_actions(index)
        else:
            current_predictor = copy.deepcopy(self.predictor)
            chosed_actions = copy(self.res.chosed_actions)

            # one run
            score = self.get_score()
            index = np.argmax(score)
            action = self.unchosed_actions(index)
            num_sampling = self.config.search.num_sampling

            for n in xrange(1, N):
                if self.config.predict.is_rand_expans:
                    current_test = variable(X=test.X[self.unchosed_actions, :],
                                            Z=test.Z[self.unchosed_actions, :])
                else:
                    temp_test = variable(X=test.X[self.unchosed_actions, :])

                t_hat = current_predictor.get_predict_samples(
                    self.training, current_test, N=num_sampling)

                score = np.zeros((num_sampling, self.test.X.shape[0]))

                for i in xrange(num_sampling):
                    self.compute_stats(current_test[i, :], t_hat[i, n], predictor=current_predictor)

                current_test = self.test.get_subset(chosed_actions)
                score[i, :] = self.get_score(current_test, predictor=current_predictor)
                index = np.argmax(np.mean(score, 0))
                action[n] = self.unchosed_actions[index]
                self.delete_unchosed_actions()

        self.unchosed_actions = self.delete_unchosed_actions(index)
        return action

    def compute_stats(self, training=None, predictor=None):
        if predictor is None:
            predictor = self.predictor

        if training is None:
            training = self.training

        new_test = variable(X=new_x, t=new_t, Z=new_z)
        try:
            predictor.update(self.training, new_test)
        except:
            predictor.prepare(self.training)

        return predictor

    def is_learning(self, n, learn_timing, num_do_learning):
        return self.config.learning.is_hyparams_learning \
                and num_do_learning < len(learn_timing) \
                and learn_timing[num_do_learning] == n

    def learning(self):
        self.predictor.fit(self.training)
        self.training.Z = self.predictor.get_basis(self.training.X)
        self.test.Z = self.predictor.get_basis(self.test.X)
        self.predictor.prepare(self.training)

    def random_search(self, max_num_probe=1,
                      num_search_each_probe=1, simu=None):

        N = int(num_search_each_probe)

        if int(max_num_probe) * N > self.test.X.shape[0]:
            raise ValueError('max_num_probe * num_search_each_probe must \
                be smaller than the length of candidates')

        if self.config.search.is_disp and simu is None \
           and self.res.total_num_search == 0:
            print 'Interaction Mode Start'

        for n in xrange(0, max_num_probe):
            if self.config.search.is_disp and N > 1:
                print '%04d-th multiple probe search (random) \n' % (n+1)

            action = self.random_action(N)

            if simu is None:
                return action

            new_t, new_x = self.call_simulator(simu, action)
            self.write(action, new_t, new_x)

            if self.config.search.is_disp:
                self.show(N)

        return self.res

    def show(self, N):
        if N == 1:
            print '%04d-th step: f(x) = %f, current best = %f, \
                   (best action = %d) \n'\
                % (self.res.total_num_search,
                   self.res.fx[self.res.total_num_search-1],
                   self.res.best_fx[self.res.total_num_search-1],
                   self.res.best_action[self.res.total_num_search-1])
        else:
            st = self.res.total_num_search - N
            en = self.res.total_num_search

            previous = self.res.multi_probe_best_fx[self.res.num_probe-2]
            current = self.res.multi_probe_best_fx[self.res.num_probe-1]
            best_action = \
                self.res.best_action[self.res.total_num_search-1]

            if current != previous:
                print 'current best = %f (best action = %d) updated !! ' \
                   % (current, best_action)
            else:
                print 'current best = %f (best action = %d)' \
                    % (self.res.multi_probe_best_fx[self.res.num_probe-1],
                       best_action)

            print 'list of simulation results'
            for n in xrange(st, en):
                print 'f(x) = %f (action = %d)' % (self.res.fx[n],
                                                   self.res.chosed_actions[n])

            print '\n'

    def set_predictor(self, predictor=None):
        self.predictor = predictor

        if self.predictor is None:
            if self.config.predict.is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)
