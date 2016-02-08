import numpy as np
from res import res

import scipy.stats
from disc_policy import disc_policy

class disc_blm_search( disc_policy ):
    def __init__( self, simu, test_X, config, train_X = None, train_t = None, score = 'TS' ):
        super(disc_blm_search, self).__init__(simu, test_X, config, train_X = None, train_t = None )
        self.score = score

    def gp_search( self, gp, num_rand_search,  max_search, timing = None ):
        num_temp = 0
        timing = self.gen_timing( num_rand_search, max_search, timing )

        if self.config.search.is_hyparams_learning and num_rand_search == timing[ num_temp ]:
            gp.fit( self.train_X, self.train_t, self.config )
            blm = gp.export_blm(self.config.predict.num_basis)
            train_Psi = blm.lik.get_basis( self.train_X )
            test_Psi = blm.lik.get_basis( self.test_X )
            blm.prepare( self.train_X, self.train_t, train_Psi )
            num_temp += 1

        for n in xrange( num_rand_search, max_search ):
            st = time.time()
            temp = self.get_action( blm, self.score, test_Psi )
            action = self.avail_action[ temp ]

            ''' simulation '''
            tmp_t, tmp_x = self.call_simu( action )

            if tmp_x is None:
                tmp_x = self.test_X[ temp, : ]
                tmp_psi = test_Psi[ temp, :]
            else:
                tmp_psi = blm.lik.get_basis(tmp_x)

            self.res.write( tmp_t, action )

            if self.is_disp:
                self.res.print_search_res( n )

            self.add_train_data( tmp_t, tmp_x )
            train_Psi = self.add_psi_data( train_Psi, tmp_psi )
            self.del_avail_action( temp )
            self.test_X = self.del_test_data( self.test_X, temp )
            test_Psi = self.del_test_data( test_Psi, temp )

            if num_temp < len( timing ) and self.config.search.is_hyparams_learning \
                and n == timing[ num_temp ]:
                gp.fit( self.train_X, self.train_t, self.config )
                blm = gp.export_blm(self.config.predict.num_basis)
                train_Psi = blm.lik.get_basis( self.train_X )
                test_Psi = blm.lik.get_basis( self.test_X )
                blm.prepare( self.train_X, self.train_t, train_Psi )
                num_temp += 1
            else:
                if isinstance(tmp_t,np.ndarray):
                    len_output = len(tmp_t)
                    for i in xrange( len_output ):
                        blm.update_stats( tmp_x[i,:], tmp_t[i],tmp_psi[i,:])
                else:
                    blm.update_stats( tmp_x, tmp_t, tmp_psi )

            self.res.search_time[n] = time.time()- st

    def add_psi_data(self, train_Psi, psi ):
        train_Psi = np.vstack( (train_Psi, psi) )
        return train_Psi

    def get_action( self, blm, score, Psi ):
        if score == 'PI':
            action = self.score_PI( blm, self.res._max_t, Psi )
        elif score == 'EI':
            action = self.score_EI( blm, self.res._max_t, Psi )
        elif score == 'TS':
            action = self.score_TS( blm, Psi )
        else:
            raise NotImplementedError

        return action

    def score_PI( self, blm, fmax, Psi ):
        post_fmean = blm.get_post_fmean( self.test_X, Psi  )
        post_fcov = blm.get_post_fcov( self.test_X, Psi  )
        temp = ( post_fmean - fmax )/np.sqrt( post_fcov )
        score = scipy.stats.norm.cdf( temp )
        return np.argmax( score )

    def score_EI( self, blm, fmax, Psi ):
        post_fmean = blm.get_post_fmean( self.test_X, Psi )
        post_fcov = blm.get_post_fcov( self.test_X, Psi )
        temp1 = ( post_fmean - fmax )
        temp2 = temp1 / np.sqrt( post_fcov )
        return np.argmax( temp1 * scipy.stats.norm.cdf( temp2 ) \
                + np.sqrt( post_fcov ) * scipy.stats.norm.pdf(temp2) )

    def score_TS( self, blm, Psi ):
        score = blm.post_sampling( self.test_X, Psi = Psi )
        return np.argmax( score )

    def gen_timing( self, num_rand_search, max_search, timing = None ):
        if timing is None:
            if self.config.search.learning_timing == 0:
                timing = num_rand_search
            else:
                timing = xrange( num_rand_search, max_search, self.config.search.learning_timing )
        return timing

    def save( self, file_name = None ):
        if file_name is None:
            if self.seed is None:
                file_name = 'bayes_search'
            else:
                file_name = 'bayes_search_%03d' %( self.seed )

        self.res.save( file_name )
