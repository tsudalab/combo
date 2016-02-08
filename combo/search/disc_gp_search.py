import numpy as np
import time
import scipy.stats
from disc_policy import disc_policy

class disc_gp_search( disc_policy ):
    def __init__( self, simu, test_X, config, train_X = None, train_t = None, score = 'PI' ):
        super(disc_gp_search, self).__init__(simu, test_X, config, train_X = None, train_t = None )
        self.score = score

    def gp_search( self, gp, num_rand_search, max_search,  timing = None ):
        num_temp = 0
        timing = self.gen_timing( num_rand_search, max_search, timing )

        for n in xrange( num_rand_search, max_search ):
            st = time.time()

            if num_temp < len( timing ) and self.config.search.is_hyparams_learning \
                and n == timing[ num_temp ]:

                gp.fit( self.train_X, self.train_t, self.config )
                num_temp += 1

            gp.prepare( self.train_X, self.train_t )
            temp = self.get_action( gp, self.score )
            action = self.avail_action[ temp ]

            ''' simulation '''
            tmp_t, tmp_x = self.call_simu( action )

            if tmp_x is None:
                tmp_x = self.test_X[ temp, : ]

            self.res.write( tmp_t, action )

            if self.is_disp:
                self.res.print_search_res( n )

            self.add_train_data( tmp_t, tmp_x )
            self.del_avail_action( temp )
            self.test_X = self.del_test_data( self.test_X, temp )
            self.res.search_time[n] = time.time()- st

    def get_action( self, gp, score ):
        if score == 'PI':
            action = self.score_PI( gp, self.res._max_t )
        elif score == 'EI':
            action = self.score_EI( gp, self.res._max_t )
        else:
            raise NotImplementedError

        return action

    def score_PI( self, gp, fmax ):
        post_fmean = gp.get_post_fmean( self.train_X, self.test_X  )
        post_fcov = gp.get_post_fmean( self.train_X, self.test_X  )
        temp = ( post_fmean - fmax )/np.sqrt( post_fcov )
        score = scipy.stats.norm.cdf( temp )
        return np.argmax( score )

    def score_EI( self, gp, fmax ):
        post_fmean = gp.get_post_fmean( self.train_X, self.test_X  )
        post_fcov = gp.get_post_fmean( self.train_X , self.test_X )
        temp1 = ( post_fmean - fmax )
        temp2 = temp1 / np.sqrt( post_fcov )
        return np.argmax( temp1 * scipy.stats.norm.cdf( temp2 ) \
                + np.sqrt( post_fcov ) * scipy.stats.norm.pdf(temp2) )

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
