import numpy as np
import time
import copy
import scipy.stats
from policy import policy

class multi_probe_policy( policy ):
    def __init__( self, simu, test_X, config ):
        super( multi_probe_policy, self ).__init__( simu, test_X, config )

    def run( self, gp, timing = None, train_X = None, train_t = None, file_name = None ):
        self.set_init_train_data( train_X, train_t)
        self.rand_search( self.config.search.num_rand_search )

        if self.config.predict.is_rand_expans:
            self.blm_search( gp, timing )
        else:
            self.gp_search(  gp, timing )
        self.save( file_name )

    def learning( self, gp, timing ):
        if self.count_learning < len(timing) and self.num_search == timing[self.count_learning]:
            gp.fit( self.train_X, self.train_t, self.config)
            self.count_learning += 1

    def gp_search( self, gp, timing = None ):
        num_multi_probe = self.config.search.num_multi_probe
        num_max_search = int(np.floor((self.config.search.max_search - self.config.search.num_rand_search)/num_multi_probe))
        num_sampling = self.config.search.multi_probe_num_sampling
        timing = self.gen_timing( timing )
        self.count_learning = 0
        self.num_search = np.copy( self.config.search.num_rand_search )

        num_dim = self.train_X.shape[1]
        for n in xrange( num_max_search ):
            if self.config.learning.is_hyparams_learning:
                self.learning( gp, timing )

            action_list = np.zeros( num_multi_probe, dtype = int )

            # compute the statistical moments
            gp.prepare( self.train_X, self.train_t )
            curr_gp = copy.deepcopy(gp)
            curr_train_X = np.copy( self.train_X )
            score = self.get_score_gp( gp, self.train_X, self.test_X, self.config.search.score )
            temp_action = np.argmax( score )
            action = self.avail_action[ temp_action ]
            action_list[ 0 ] = action
            curr_test_X = self.test_X[ temp_action, :]
            self.train_X = np.vstack( (self.train_X, self.test_X[ temp_action, : ]) )
            self.del_avail_action( temp_action )
            self.test_X = self.del_data( self.test_X, temp_action )

            for k in xrange( 1, num_multi_probe ):
                # generates the psuedo outcomes from the current predictive distribution
                t_hat = curr_gp.predict_sampling( curr_train_X, curr_test_X.reshape( (k, num_dim) ), N = num_sampling )

                # computation of the score for searching
                score = np.zeros( (num_sampling, self.test_X.shape[0]) )
                for i in xrange( num_sampling ):
                    virtual_t = np.hstack( ( np.copy( self.train_t ), t_hat[i,:] ) )
                    gp.prepare( self.train_X, virtual_t )
                    score[i,:] = self.get_score_gp( gp, self.train_X, self.test_X, self.config.search.score )

                temp_action = np.argmax( np.mean( score, 0 ) )

                action = self.avail_action[ temp_action ]
                action_list[ k ] = action

                self.train_X = np.vstack( (self.train_X, self.test_X[ temp_action, : ]) )
                curr_test_X = np.vstack( ( curr_test_X, self.test_X[ temp_action, : ] ) )
                self.del_avail_action( temp_action )
                self.test_X = self.del_data( self.test_X, temp_action )

            train_t = self.simu( action_list )
            self.train_t = np.hstack( ( self.train_t, train_t ) )
            print '%02d-th multiple probe search' %(n)
            self.write( num_multi_probe, train_t, action_list )


    def blm_search( self, gp, timing = None ):
        num_multi_probe = self.config.search.num_multi_probe
        num_max_search = int(np.floor((self.config.search.max_search - self.config.search.num_rand_search)/num_multi_probe))
        num_sampling = self.config.search.multi_probe_num_sampling
        self.count_learning = 0
        self.num_search = np.copy( self.config.search.num_rand_search )

        timing = self.gen_timing( timing )

        for n in xrange( num_max_search ):
            if self.config.learning.is_hyparams_learning:
                self.learning( gp, timing )
                blm = gp.export_blm( self.config.predict.num_basis )
                self.train_Psi = blm.lik.get_basis( self.train_X )
                self.test_Psi = blm.lik.get_basis( self.test_X )
            elif n == 0:
                blm = gp.export_blm( self.config.predict.num_basis )
                self.train_Psi = blm.lik.get_basis( self.train_X )
                self.test_Psi = blm.lik.get_basis( self.test_X )
            else:
                N = self.train_X.shape[0]
                for k in xrange(num_multi_probe):
                    blm.update_stats(self.train_X[N-k-1,:], self.train_t[N-k-1], self.Psi[N-k-1, :] )

            action_list = np.zeros( num_multi_probe, dtype = int )
            blm.prepare( self.train_X, self.train_t, self.train_Psi )
            curr_blm = copy.deepcopy( blm )

            curr_train_X = np.copy( self.train_X )
            curr_train_Psi = np.copy( self.train_Psi )

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

    def get_score_blm( self, blm, score ):
        if score == 'PI':
            post_fmean = blm.get_post_fmean( self.test_X, self.test_Psi  )
            post_fstd = np.sqrt( blm.get_post_fcov( self.test_X, self.test_Psi  ) )
            fmax = np.max( blm.get_post_fmean( self.train_X, self.train_Psi  ) )
            temp = ( post_fmean - fmax )/post_fstd
            score = scipy.stats.norm.cdf( temp )
        elif score == 'EI':
            post_fmean = blm.get_post_fmean( self.test_X, self.test_Psi  )
            post_fstd = np.sqrt( blm.get_post_fcov( self.test_X, self.test_Psi  ) )
            fmax = np.max( blm.get_post_fmean( self.train_X, self.train_Psi  ) )
            temp1 = ( post_fmean - fmax )
            temp2 = temp1 / post_fstd
            score = temp1 * scipy.stats.norm.cdf( temp2 ) + post_fstd * scipy.stats.norm.pdf(temp2)
        elif score == 'TS':
            score = blm.post_sampling( self.test_X, Psi = self.test_Psi, alpha = self.config.search.alpha )
        else:
            raise NotImplementedError
        return score


    def get_score_gp( self, gp, train_X, test_X, score_name ):
        post_fmean = gp.get_post_fmean( train_X, test_X  )
        post_fstd = np.sqrt( gp.get_post_fcov( train_X, test_X  ) )
        fmax = np.max( gp.get_post_fmean( train_X, train_X ) )
        #fmax = np.max(post_fmean) #self.res._max_t
        if score_name == 'PI':
            temp = ( post_fmean - fmax )/post_fstd
            score = scipy.stats.norm.cdf( temp )
        elif score_name == 'EI':
            temp1 = ( post_fmean - fmax )
            temp2 = temp1 / post_fstd
            score = temp1 * scipy.stats.norm.cdf( temp2 ) + post_fstd * scipy.stats.norm.pdf(temp2)
        else:
            raise NotImplementedError
        return score


    def gen_timing( self, num_multi_probe, timing = None ):
        if timing is None and self.config.learning.is_hyparams_learning:
            if self.config.learning.interval == 0:
                timing = np.array([self.config.search.num_rand_search])
            else:
                if num_multi_probe  > self.config.learning.interval:
                    self.config.learning.interval = num_multi_probe
                else:
                    self.config.learning.interval = int(np.floor(self.config.learning.interval / self.config.search.num_multi_probe)) * self.config.search.num_multi_probe

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


    def write( self, num_multi_probe, t, action_list ):
        for k in xrange( num_multi_probe ):
            self.res.write( t[k], action_list[k] )

            if self.config.search.is_disp:
                self.res.print_search_res( self.num_search )

            self.num_search += 1
