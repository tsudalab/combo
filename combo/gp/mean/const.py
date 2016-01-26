import numpy as np

class const:
    ''' constant '''
    def __init__( self, params = None ):
        self._init_params(params)
        self.nparams = 1

    def get_mean( self, ndata, params = None ):
        ''' compute the mean of prior '''
        if params is None:
            params = np.copy(self.params)
        return params * np.ones( ndata )

    def get_grad( self, ndata, params = None ):
        ''' derive the gradient with respect to the parameter '''
        return np.ones(ndata)

    def set_params( self, params ):
        self.params = params

    def show( self ):
        print 'nparams = %d' %( self.nparams )
        print 'params  = ', self.params

    def _init_params( self, params ):
        ''' initilize < params >'''
        if params is None:
            self.params = 0
        else:
            self.params = params
