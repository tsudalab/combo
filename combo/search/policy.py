import numpy as np
import time
from res import res

class policy( object ):
    def __init__( self, simu, test_X, config ):
        self.simu = simu
        self.seed = None
        self.test_X = test_X
        self.num_dim = self.test_X.shape[1]
        self.config = config
        self.avail_action = range( 0, self.test_X.shape[0] )
        self.res = res( config )

    def set_init_train_data( self, train_X = None, train_t = None ):
        self.train_X = train_X
        self.train_t = train_t

        if self.train_X is None and self.train_t is None:
            self.train_X = np.empty( ( 0, self.num_dim ) )
            self.train_t = np.empty( 0 )

    def set_seed( self, seed ):
        self.seed = seed
        np.random.seed( seed )

    def add_data( self, t, x ):
        self.train_t = np.hstack( (self.train_t, t) )
        self.train_X = np.vstack( (self.train_X, x) )

    def del_data( self, X, num_row ):
        return np.delete( X, num_row,0 )

    def del_avail_action( self, num_row ):
        self.avail_action.pop( num_row )

    def call_simu( self, action ):
        output = self.simu( action )
        if hasattr(output,'__len__'):
            t = output[0]; x = output[1]
        else:
            t = output; x = None

        return t, x

    def rand_search( self, num_rand_search ):
        for n in xrange( 0, num_rand_search ):
            st_search_time = time.time()
            itemp = np.random.randint( 0, self.test_X.shape[0] )
            action = self.avail_action[ itemp ]

            # calling the simulator
            tmp_t, tmp_x = self.call_simu( action )

            if tmp_x is None:
                tmp_x = self.test_X[itemp,:]

            self.res.write( tmp_t, action )

            if self.config.search.is_disp:
                self.res.print_search_res( n )

            self.add_data( tmp_t, tmp_x )
            self.del_avail_action( itemp )
            self.test_X = self.del_data( self.test_X, itemp )
            self.res.search_time[n] = time.time() - st_search_time
