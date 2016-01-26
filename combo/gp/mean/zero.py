import numpy as np

class zero:
    ''' zero '''
    def __init__(self):
        self.nparams = 0
        self.params = np.array([])

    def get_mean(self, ndata, params = None):
        return np.zeros(ndata)

    def get_grad( self, ndata, params = None):
        return np.array([])

    def set_params( self, params ):
        pass

    def show( self ):
        print ('nparams = %d') %(self.nparams)
        print 'params  = ', self.params
