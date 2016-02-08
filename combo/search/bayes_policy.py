import numpy as np
import scipy.stats
import time
from policy import policy

class bayes_policy( policy ):
    def __init__( self, simu, test_X, config ):
        super( bayes_policy, self).__init__( simu, test_X, config )

    def run( self, gp, timing = None, train_X = None, train_t = None, file_name = None ):
        self.set_init_train_data( train_X, train_t)
        print 'start the random search .....\n'
        self.rand_search( self.config.search.num_rand_search )

        print 'start the bayes search ....\n'
        if self.config.predict.is_rand_expans:
            self.blm_search( gp, timing )
        else:
            self.gp_search(  gp, timing )

        self.save( file_name )

    def blm_search( self, gp, timing = None ):
        timing = self.gen_timing( timing )
        num_temp = 0

        n = self.config.search.num_rand_search
        if self.config.learning.is_hyparams_learning:
            if num_temp < len(timing) and self.config.search.num_rand_search == timing[num_temp]:
                st_learn_time = time.time()
                gp.fit( self.train_X, self.train_t, self.config)
                self.res.learning_time[n] = time.time() - st_learn_time

                st_infer_time = time.time()
                blm = gp.export_blm( self.config.predict.num_basis )
                self.train_Psi = blm.lik.get_basis( self.train_X )
                self.test_Psi = blm.lik.get_basis( self.test_X )
                blm.prepare( self.train_X, self.train_t, self.train_Psi )
                self.res.infer_time[n] = time.time() - st_infer_time
                num_temp += 1

        for n in xrange( self.config.search.num_rand_search, self.config.search.max_search ):
            st_full_time = time.time()

            st_search_time = time.time()
            temp = self.get_action_blm( blm, self.config.search.score )
            self.res.search_time[n] = time.time()- st_search_time

            action = self.avail_action[ temp ]

            st_simu_time = time.time()
            tmp_t, tmp_x = self.call_simu( action )
            self.res.simu_time[n] = time.time() - st_simu_time

            if tmp_x is None:
                tmp_x = self.test_X[ temp, : ]
                tmp_psi = self.test_Psi[ temp, :]
            else:
                tmp_psi = blm.lik.get_basis(tmp_x)

            self.res.write( tmp_t, action )

            if self.config.search.is_disp:
                self.res.print_search_res( n )

            self.add_data( tmp_t, tmp_x )
            self.train_Psi = np.vstack((self.train_Psi, tmp_psi))
            self.del_avail_action( temp )
            self.test_X = self.del_data( self.test_X, temp )
            self.test_Psi = self.del_data( self.test_Psi, temp )

            if self.config.learning.is_hyparams_learning and num_temp < len(timing) and n == timing[num_temp]:
                st_learn_time = time.time()
                gp.fit( self.train_X, self.train_t, self.config)
                self.res.learning_time[n] = time.time() - st_learn_time

                st_infer_time = time.time()
                blm = gp.export_blm(self.config.predict.num_basis)
                self.train_Psi = blm.lik.get_basis( self.train_X )
                self.test_Psi = blm.lik.get_basis( self.test_X )
                blm.prepare( self.train_X, self.train_t, self.train_Psi )
                self.res.infer_time[n] = time.time() - st_simu_time
                num_temp += 1
            else:
                st_infer_time = time.time()
                if hasattr(tmp_t,'__len__'):
                    for i in xrange(len(tmp_t)):
                        blm.update_stats(tmp_x[i,:], tmp_t[i], tmp_psi[i,:])
                else:
                    blm.update_stats(tmp_x, tmp_t, tmp_psi)

                self.res.infer_time[n] = time.time() - st_infer_time

            self.res.full_time[n] = time.time() - st_full_time

    def gp_search( self, gp, timing = None ):
        timing = self.gen_timing( timing )
        num_temp = 0

        for n in xrange( self.config.search.num_rand_search, self.config.search.max_search ):
            st_full_time = time.time()

            if self.config.learning.is_hyparams_learning:
                if num_temp < len(timing) and n == timing[num_temp]:
                    st_learn_time = time.time()
                    gp.fit( self.train_X, self.train_t, self.config)
                    self.res.learning_time[n] = time.time() - st_learn_time
                    num_temp += 1

            st_infer_time = time.time()
            gp.prepare( self.train_X, self.train_t )
            self.res.infer_time[n] = time.time() - st_infer_time

            st_search_time = time.time()
            temp = self.get_action_gp( gp, self.config.search.score )
            self.res.search_time[n] = time.time()- st_search_time

            action = self.avail_action[ temp ]

            st_simu_time = time.time()
            tmp_t, tmp_x = self.call_simu( action )
            self.res.simu_time[n] = time.time() - st_simu_time

            if tmp_x is None:
               tmp_x = self.test_X[ temp, : ]

            self.res.write( tmp_t, action )

            if self.config.search.is_disp:
                self.res.print_search_res( n )

            self.add_data( tmp_t, tmp_x )
            self.del_avail_action( temp )
            self.test_X = self.del_data( self.test_X, temp )

            self.res.full_time[n] = time.time() - st_full_time

    def get_action_gp( self, gp, score ):
        post_fmean = gp.get_post_fmean( self.train_X, self.test_X  )
        post_fstd = np.sqrt( gp.get_post_fcov( self.train_X, self.test_X  ) )
        fmax = self.res._max_t
        if score == 'PI':
            temp = ( post_fmean - fmax )/post_fstd
            score = scipy.stats.norm.cdf( temp )
            action = np.argmax( score )
        elif score == 'EI':
            temp1 = ( post_fmean - fmax )
            temp2 = temp1 / post_fstd
            score = temp1 * scipy.stats.norm.cdf( temp2 ) + post_fstd * scipy.stats.norm.pdf(temp2)
            action = np.argmax(score)
        else:
            raise NotImplementedError
        return action

    def get_action_blm( self, blm, score ):
        if score == 'PI':
            post_fmean = blm.get_post_fmean( self.test_X, self.test_Psi  )
            post_fstd = np.sqrt( blm.get_post_fcov( self.test_X, self.test_Psi  ) )
            fmax = self.res._max_t
            temp = ( post_fmean - fmax )/post_fstd
            score = scipy.stats.norm.cdf( temp )
            action = np.argmax( score )
        elif score == 'EI':
            post_fmean = blm.get_post_fmean( self.test_X, self.test_Psi  )
            post_fstd = np.sqrt( blm.get_post_fcov( self.test_X, self.test_Psi  ) )
            fmax = self.res._max_t
            temp1 = ( post_fmean - fmax )
            temp2 = temp1 / post_fstd
            score = temp1 * scipy.stats.norm.cdf( temp2 ) + post_fstd * scipy.stats.norm.pdf(temp2)
            action = np.argmax(score)
        elif score == 'TS':
            score = blm.post_sampling( self.test_X, Psi = self.test_Psi )
            action = np.argmax( score )
        else:
            raise NotImplementedError
        return action


    def gen_timing( self, timing = None ):
        if timing is None and self.config.learning.is_hyparams_learning:
            if self.config.learning.interval == 0:
                timing = np.array([self.config.search.num_rand_search])
            else:
                timing = xrange(self.config.search.num_rand_search, self.config.search.max_search, self.config.learning.interval)

        if self.config.learning.is_hyparams_learning is False:
            timing = np.zeros(1)
        return timing


    def save(self, file_name = None ):
        if file_name is None:
            if self.seed is None:
                file_name = 'bayes_search'
            else:
                file_name = 'bayes_search_%03d' %( self.seed )
        self.res.save( file_name )
