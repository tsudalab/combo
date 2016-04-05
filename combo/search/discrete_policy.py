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
        self.max_fx = np.zeros(max_length_search, dtype=float)
        self.max_fx_each_probe = np.zeros(max_length_search, dtype=float)
        self.chosed_actions = np.zeros(max_length_search, dtype=int)

    def write(self, t, action, N):
        st = self.total_num_search
        en = st + N

        for n in xrange(st, en):
            self.fx[n] = t if N == 1 else t[n]

            if n == 0:
                self.max_fx[n] = t if N == 1 else t[n]
            else:
                self.max_fx[n] = t if N == 1 else t[n]

            self.chosed_actions[n] = action if N == 1 else action[n]

        max_t = t if N == 1 else np.max(t)
        k = self.num_probe
        if k == 0:
            self.max_fx_each_probe[k] = max_t
        else:
            tmp = self.max_fx_each_probe[k-1]
            self.max_fx_each_probe[k] = max_t if tmp < max_t else tmp

        self.total_num_search = en
        self.num_probe += 1

    def save(self, filename):
        N = self.total_num_search
        M = self.num_probe
        np.savez_compressed(filename, num_probe=M, total_num_search=N,
                            fx=self.fx[0:N],
                            max_fx=self.max_fx[0:N],
                            max_fx_each_probe=self.max_fx_each_probe[0:M],
                            chosed_actions=self.chosed_actions[0:N])

    def load(self, filename):
        data = np.load(filename)
        self.num_probe = data['num_probe']
        self.total_num_search = data['total_num_search']
        N = self.total_num_search
        self.fx[0:N] = data['fx']
        self.max_fx[0:N] = data['max_fx']
        self.max_fx_each_probe[0:N] = data['max_fx_each_probe']
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

    def load(self, filename, training=None):
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

    def call_simulator(self, simu, action):
        output = simu(action)
        if hasattr(output, '__len__') and len(output) == 2:
            t = output[0]
            x = output[1]
        else:
            t = output
            x = self.test.X[action, :]
        return t, x

    def write(self, new_x, new_t, new_z=None):
        if hasattr(new_t, '__len__'):
            N = len(new_t)
        else:
            N = 1
        self.res.write(new_x, new_t, N)
        self.training.add(new_x, new_t, new_z)

    def random_search(self, max_num_probe, num_search_each_probe=1, simu=None):
        N = int(num_search_each_probe)

        if int(max_num_probe) * N > self.test.X.shape[0]:
            raise ValueError('max_num_probe * num_search_each_probe must \
                be smaller than the length of candidates')

        for n in xrange(0, max_num_probe):
            if self.config.search.is_disp and N > 1:
                    print '%04d-th multiple probe search (random) \n' % (n+1)

            random_index \
                = np.random.permutation(xrange(self.unchosed_actions.shape[0]))

            index = random_index[0:N]
            action = self.unchosed_actions[index]
            self.delete_unchosed_actions(index)

            if simu is None:
                return action

            new_t, new_x = self.call_simulator(simu, action)

            self.write(new_x, new_t)

            if self.config.search.is_disp:
                self.show(N)

        return self.res


    def show(self, N=1):
        if N == 1:
            print '%04d-th step: f(x) = %f, max_f(x) = %f \n'\
                % (self.res.total_num_search,
                   self.res.fx[self.res.total_num_search-1],
                   self.res.max_fx[self.res.total_num_search-1])
        else:
            st = self.res.total_num_search - N
            en = self.res.total_num_search
            for n in xrange(st, en):
                print 'f(x) = %f, max_f(x) = %f \n' \
                        % (self.res.fx[n], self.res.max_fx[n])

            print '\n'
            print '%04d-th multiple probe search: max_f = %f ' \
                % (self.res.num_search,
                   self.res.max_fx_each_probe[self.res.num_probe-1])

    '''
    def set_predictor(self, predictor=None):
        self.predictor = predictor

        if self.predictor is None:
            if self.config.predict.is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)'''

'''
class search_results:
    def __init__(self, dir_name='res', file_name='results', save_interval=0):
        self.num_search = 0
        self.fx = np.zeros(max_len_search, dtype=float)
        self.max_fx = np.zeros(max_len_search, dtype=float)
        self.action_log = np.zeros(max_len_search, dtype=int)
        self.save_interval = save_interval
        self.file_name = file_name
        self.dir_name = dir_name
        self.make_dir(dir_name)

    def make_dir(self, dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        dir_name = str(dir_name) + '/temp'

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

    def write(self, t, action):
        self.fx[self.num_search] = t
        if self.num_search == 0:
            self.max_fx[0] = t
        else:
            self.max_fx[self.num_search] \
                = max(t, self.max_fx[self.num_search-1])
        self.action_log[self.num_search] = action

        if self.num_search > 0 \
            and self.save_interval > 0 \
                and np.mod(self.num_search, self.save_interval) == 0:

            file_name = str(self.dir_name) + '/temp/' \
                + str(self.file_name) + '.npz'
            self.save(file_name)

        self.num_search += 1

    def save(self, file_name):
        with open(str(file_name), 'w') as f:
            np.savez_compressed(f, fx=self.fx[0:self.num_search],
                                max_fx=self.max_fx[0:self.num_search],
                                action_log=self.action_log[0:self.num_search])


class policy(object):
    def __init__(self, config, test, dir_name='res', predictor=None):
        self.config = config
        self.predictor = predictor
        self.dir_name = dir_name
        self.training = variable()

        if isinstance(test, np.ndarray):
            self.test = variable(X=test)
        elif isinstance(test, variable):
            self.test = test
        else:
            raise TypeError('test must take ndarray or combo.variable')

        self.set_predictor(predictor)
        self.action_set = range(0, self.test.X.shape[0])

    def set_predictor(self, predictor=None):
        self.predictor = predictor

        if self.predictor is None:
            if self.config.predict.is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)

    def delete_action_set(self, num_row):
        self.action_set.pop(num_row)

    def call_simulator(self, simulator, action):
        output = simulator(action)
        if hasattr(output, '__len__'):
            t = output[0]
            x = output[1]
        else:
            t = output
            x = None
        return t, x

    def random_search(self, max_search, simulator=None, search_res=None):
        if search_res is None:
            search_res = search_results(self.dir_name)

        st_num_search = search_res.num_search

        for n in xrange(st_num_search, max_search):
            action, num_row = self.get_action(is_random=True)
            new_t, new_x = self.call_simulator(simulator, action)

            if new_x is None:
                new_x = self.test.X[num_row, :]

            search_res.write(new_t, action)

            if self.config.search.is_disp:
                self.show(search_res)

            self.training.add(new_x, new_t)
            self.test.delete_x(num_row)
            self.test.add_t(new_t)

        return search_res

    def show(self, search_res):
        print '%04d-th step: f(x) = %f, max_f(x) = %f \n'\
            % (search_res.num_search,
               search_res.fx[search_res.num_search-1],
               search_res.max_fx[search_res.num_search-1])

    def get_learn_timinig(self, learn_timing=None):
        if self.config.learning.is_hyparams_learning:
            if learn_timing is None:
                if self.config.learning.interval == 0:
                    learn_timing = np.array([0])
                else:
                    learn_timing =\
                        np.arange(0,
                                  self.config.search.max_search,
                                  self.config.learning.interval)
        else:
            learn_timing = np.array([-1])
        return learn_timing

    def get_score(self, predictor=None):
        if predictor is None:
            predictor = self.predictor

        if self.config.search.score == 'EI':
            f = score.EI(predictor, self.training, self.test)
        elif self.config.search.score == 'PI':
            f = score.PI(predictor, self.training, self.test)
        elif self.config.search.score == 'TS':
            f = score.TS(predictor, self.training, self.test,
                         alpha=self.config.search.alpha)
        else:
            raise NotImplementedError

        return score

    def get_action(self, is_random=False):
        if is_random:
            num_row = np.random.randint(0, self.test.X.shape[0])
        else:
            f = self.get_score()

            num_row = np.argmax(f)

        action = self.action_set[num_row]
        self.delete_action_set(num_row)

        return action, num_row

    def save(self, file_name):
        with open(file_name, 'w') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, file_name):
        with open(file_name) as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def learning(self):
        self.predictor.fit(self.training)
        self.training.Z = self.predictor.get_basis(self.training.X)
        self.test.Z = self.predictor.get_basis(self.test.X)
        self.predictor.prepare(self.training)

    def multiple_bayes_search(self, simulator, max_search, training=None,
                              search_res=None, learn_timing=None,
                              num_multi_probe=10):
        if search_res is None:
            search_res = search_results(self.dir_name)

        if training is not None:
            self.training = training

        learn_timing = self.get_learn_timinig(learn_timing)
        num_do_learning = 0

        if self.config.learning.is_hyparams_learning \
            and num_do_learning < len(learn_timing) \
                and learn_timing[num_do_learning] == 0:
            self.learning()
            num_do_learning += 1

        for n in xrange(0, max_search):




    def get_multi_action(self, num_multi_probe):
        current_predictor = copy.deepcopy(self.predictor)
        current_training = copy.deepcopy(self.training)
        action, num_row = get_action()
        new_x = self.test.X[num_row, :]

        if test.Z is None:
            new_z = None
        else:
            new_z = self.test.Z[num_row, :]

        current_test = variable(X=self.test.X[num_row, :],
                                Z=self.test.Z[num_row, :])

        self.training.add_x(new_x)
        self.training.add_z(new_z)
        self.test.delete_x(num_row)
        self.test.delete_z(num_row)

        num_sampling = self.config.search.multi_probe_num_sampling
        action_list = np.zeros(num_multi_probe)
        action_list[0] = action

        for k in xrange(1, num_multi_probe):
            t_hat = self.predictor.get_predict_samples(self.training,
                                                       current_test,
                                                       N=num_sampling)

            score = np.zeros((num_sampling, self.test.X.shape[0]))

            for i in xrange(num_sampling):
                model = copy.deepcopy(self.predictor)
                try:
                    model.update(training, current_test, t_hat[:, i])
                except:
                    model.prepare(training)

                score[i, :] = self.get_score(model)

            num_row = np.argmax(np.mean(score, 0))
            action = self.action_set[num_row]
            action_list[k] = action

            self.training.add_x(new_x=self.test.X[num_row, :])
            self.training.add_z(new_z=self.test.Z[num_row, :])
            self.delete_action_set(num_row)
            self.test.delete_x(num_row)
            self.test.delete_z(num_row)

    return action_list

'''
#curr_train_X = np.copy( self.train_X )
#curr_train_Psi = np.copy( self.train_Psi )

'''
score = self.get_score_blm( blm, self.config.search.score )
temp_action = np.argmax( score )
action = self.avail_action[ temp_action ]
action_list[ 0 ] = action

curr_test_X = self.test_X[ temp_action, :]
curr_test_Psi = self.test_Psi[ temp_action, :]

self.train_X = np.vstack( (self.train_X, self.test_X[ temp_action, : ]) )
self.train_Psi = np.vstack( (self.train_Psi, self.test_Psi[ temp_action, : ]) )
self.del_avail_action( temp_action )
self.test_X = self.del_data( self.test_X, temp_action )
self.test_Psi = self.del_data( self.test_Psi, temp_action )


for k in xrange( 1, num_multi_probe ):
    print 'k = %d' %(k)
    w_hat = curr_blm.sampling( N = num_sampling ) # D * N
    Psi = curr_test_Psi.reshape( ( k, self.config.predict.num_basis ) )
    t_hat = np.dot( Psi, w_hat ) + np.sqrt( blm.lik.cov.sigma2 ) * np.random.randn( k, num_sampling )
    # K * N

    # computation of the score for searching
    score = np.zeros( (num_sampling, self.test_X.shape[0]) )

    temp_curr_test_X = curr_test_X.reshape((k,self.train_X.shape[1]))
    temp_curr_test_Psi = curr_test_Psi.reshape((k,self.train_Psi.shape[1]))

    for i in xrange( num_sampling ):
        blm = copy.deepcopy( curr_blm )
        for j in xrange(k):
            blm.update_stats(temp_curr_test_X[j,:], t_hat[j,i], temp_curr_test_Psi[j,:])

        score[i,:] = self.get_score_blm( blm, self.config.search.score )

    temp_action = np.argmax( np.mean( score, 0 ) )
    action = self.avail_action[ temp_action ]
    action_list[ k ] = action

    self.train_X = np.vstack( (self.train_X, self.test_X[ temp_action, : ]) )
    self.train_Psi = np.vstack( (self.train_Psi, self.test_Psi[ temp_action, : ]) )
    curr_test_X = np.vstack( ( curr_test_X, self.test_X[ temp_action, : ] ) )
    curr_test_Psi = np.vstack( ( curr_test_Psi, self.test_Psi[ temp_action, : ] ) )
    self.del_avail_action( temp_action )
    self.test_X = self.del_data( self.test_X, temp_action )
    self.test_Psi = self.del_data( self.test_Psi, temp_action )

train_t = self.simu( action_list )
self.train_t = np.hstack( ( self.train_t, train_t ) )
print '%02d-th multiple probe search' %(n)
self.write( num_multi_probe, train_t, action_list )

            for n in xrange(max_search):
                action, num_row = self.get_action()
                new_t, new_x = self.call_simulator(simulator, action)

                if new_x is None:
                    new_x = self.test.X[num_row, :]

                new_z = self.predictor.get_basis(new_x)

                search_res.write(new_t, action)
                self.show(search_res)

                self.training.add(new_x, new_t, new_z)
                self.test.delete_x(num_row)
                self.test.delete_z(num_row)
                self.test.add_t(new_t)

                if self.config.learning.is_hyparams_learning \
                    and num_do_learning < len(learn_timing) \
                        and learn_timing[num_do_learning] == n:
                    self.learning()
                else:
                    try:
                        new_test = variable(X=new_x, t=new_t, Z=new_z)
                        self.predictor.update(training, new_test)
                    except:
                        self.predictor.prepare(self.training)

            return search_res
'''

'''
    def bayes_search(self, simulator, max_search, training=None,
                     search_res=None, learn_timing=None):

        if search_res is None:
            search_res = search_results(self.dir_name)

        if training is not None:
            self.training = training

        learn_timing = self.get_learn_timinig(learn_timing)
        num_do_learning = 0

        if self.config.learning.is_hyparams_learning \
            and num_do_learning < len(learn_timing) \
                and learn_timing[num_do_learning] == 0:

            self.learning()
            num_do_learning += 1

        for n in xrange(max_search):
            action, num_row = self.get_action()
            new_t, new_x = self.call_simulator(simulator, action)

            if new_x is None:
                new_x = self.test.X[num_row, :]

            new_z = self.predictor.get_basis(new_x)

            search_res.write(new_t, action)
            self.show(search_res)

            self.training.add(new_x, new_t, new_z)
            self.test.delete_x(num_row)
            self.test.delete_z(num_row)
            self.test.add_t(new_t)

            if self.config.learning.is_hyparams_learning \
                and num_do_learning < len(learn_timing) \
                    and learn_timing[num_do_learning] == n:
                self.learning()
            else:
                try:
                    new_test = variable(X=new_x, t=new_t, Z=new_z)
                    self.predictor.update(training, new_test)
                except:
                    self.predictor.prepare(self.training)

        return search_res
'''
