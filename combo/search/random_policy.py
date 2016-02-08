import numpy as np
from policy import policy

class random_policy( policy ):
    def __init__( self, simu, test_X, config ):
        super( random_policy, self).__init__( simu, test_X, config )

    def run( self, file_name = None ):
        self.set_init_train_data()
        self.rand_search( self.config.search.max_search )
        self.save( file_name )

    def save(self, file_name = None ):
        if file_name is None:
            if self.seed is None:
                file_name = 'random_search'
            else:
                file_name = 'random_search_%03d' %( self.seed )
        self.res.save( file_name )
