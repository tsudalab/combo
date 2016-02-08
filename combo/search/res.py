import cPickle as pickle
import os
import numpy as np

class res:
    def __init__( self, config ):
        self.config = config
        self._max_search = config.search.max_search
        self._num_search_step = 0
        self._max_t = -np.inf
        self.max_fx = np.zeros( self._max_search )
        self.search_time = np.zeros( self._max_search )
        self.learning_time = np.zeros( self._max_search )
        self.infer_time = np.zeros( self._max_search )
        self.full_time = np.zeros( self._max_search )
        self.simu_time = np.zeros( self._max_search )
        self.fx = np.zeros( self._max_search )
        self.history_action = np.zeros( self._max_search )
        self.dir_name = config.search.dir_name
        self.make_dir( config )

    def make_dir( self, config ):
        ''' make the directory which memorizes the search results '''
        if not os.path.isdir( self.dir_name ):
            os.mkdir( self.dir_name )

    def write( self, t, action ):
        self.fx[ self._num_search_step ] = t
        self.history_action[ self._num_search_step ] = action

        if t > self._max_t:
            self._max_t = t

        self.max_fx[ self._num_search_step ] = self._max_t
        self._num_search_step += 1

    def print_search_res( self, num_search ):
        print '%03d-th step: f(x) = %f, max_f(x) = %f \n' \
                        %( num_search +1, self.fx[num_search], self.max_fx[num_search] )

    def save( self, file_name ):
        dir_name = self.config.search.dir_name
        file_name = dir_name + '/' + file_name + '.dump'
        with open( file_name, 'w' ) as f:
            pickle.dump( self, f )
