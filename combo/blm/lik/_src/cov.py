import numpy as np

class cov:
    def __init__( self, params = None ):
        self.params = params
        if self.params is None:
            self.params = np.log(1)
        self.nparams = 1
        self.sigma2, self.prec = self._trans_params( params )

    def get_cov( self, N, params = None ):
        ''' compute the covariance of prior '''
        if params is None:
            params = self.params

        sigma2, prec = self._trans_params( params )
        return np.identity( N ) * sigma2

    def get_prec( self, N, params = None ):
        ''' compute the precision of prior '''
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( N ) * prec

    def set_params( self, params ):
        ''' substitute the parameter into the variable <params>. '''
        self.params = params
        self.sigma2, self.prec = self._trans_params( params )

    def _trans_params( self, params = None ):
        ''' transform the parameter into variance and precision  '''
        if params is None:
            params = np.copy(self.params)

        sigma2 = np.exp( 2 * params )
        prec = 1/sigma2
        return sigma2, prec
