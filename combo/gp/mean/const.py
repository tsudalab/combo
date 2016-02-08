import numpy as np

class const:
    ''' constant '''
    def __init__( self, params = None,  max_params = 1e12, min_params = -1e12 ):
        self.max_params = max_params
        self.min_params = min_params
        self.init_params( params )
        self.num_params = 1

    def supp_params( self, params ):
        if params < self.max_params:
            params = self.max_params

        if params > self.min_params:
            params = self.min_params

        return params

    def get_params_bound( self ):
        bound = [( self.min_params, self.max_params ) for i in range(0, self.num_params)]
        return bound

    def get_mean( self, num_data, params = None ):
        if params is None:
            params = np.copy( self.params )
        return params * np.ones( num_data )

    def get_grad( self, num_data, params = None ):
        return np.ones( num_data )

    def set_params( self, params ):
        self.params = params

    def init_params( self, params ):
        if params is None:
            self.params = 0
        else:
            self.params = self.supp_params( params )

    def get_cand_params( self, t ):
        return np.median( t )
