import numpy as np
from res import res

class disc_policy( object ):
    def __init__( self, simu, test_X, config, train_X = None, train_t = None ):
        self.config = config
        self.simu = simu
        self.test_X = test_X
        self.train_X = train_X
        self.train_t = train_t
        self.avail_action = range( 0, test_X.shape[0] )
        self.res = res( config )
        self.is_disp = config.search.is_disp

        if train_X is None and train_t is None:
            self.train_X = np.empty( ( 0, test_X.shape[1] ) )
            self.train_t = np.empty( 0 )

    def set_seed( self, seed ):
        self.seed = seed
        np.random.seed( seed )

    def run(self, gp, num_rand_search, max_search, file_name = None ):
        self.rand_search(num_rand_search)
        self.gp_search( gp, num_rand_search, max_search )
        self.save( file_name )

    def rand_search( self, num_rand_search ):
        ''' random search '''
        for n in xrange( 0, num_rand_search ):
            tmp = np.random.randint( 0, self.test_X.shape[0] )
            action = self.avail_action[ tmp ]

            # call the simulator
            tmp_t, tmp_x = self.call_simu( action )

            if tmp_x is None:
                tmp_x = self.test_X[tmp,:]

            # memorize the simulation results
            self.res.write( tmp_t, action )

            if self.is_disp:
                self.res.print_search_res( n )

            self.add_train_data( tmp_t, tmp_x )
            self.del_avail_action( tmp )
            self.test_X = self.del_test_data( self.test_X, tmp )

    def add_train_data( self, t, x ):
        self.train_t = np.hstack( (self.train_t, t) )
        self.train_X = np.vstack( (self.train_X, x) )

    def del_avail_action( self, num_row ):
        self.avail_action.pop( num_row )

    def del_test_data( self, test_X, tmp_row ):
        return np.delete( test_X, tmp_row, 0 )

    def call_simu( self, action ):
        t, x = self.simu( action )
        return t, x

class disc_rand_search( disc_policy ):
    def __init__( self, simu, test_X, config ):
        super( disc_rand_search, self).__init__( simu, test_X, config )

    def run( self, num_rand_search, file_name = None ):
        self.rand_search( num_rand_search )
        self.save(file_name)

    def save(self, file_name = None ):
        if file_name is None:
            if self.seed is None:
                file_name = 'random_search'
            else:
                file_name = 'random_search_%03d' %( self.seed )

        self.res.save( file_name )
