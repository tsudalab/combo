import numpy as np

class cov_const:
    ''' isotoropic covariance matrix '''
    def __init__( self, params = None ):
        if params is None:
            self.params = np.log(1)
        self.sigma2, self.prec = self._trans_params( params )

    def get_cov( self, nbasis, params = None ):
        ''' compute the covariance of prior '''
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( nbasis ) * sigma2

    def get_prec( self, nbasis, params = None ):
        ''' compute the precision of prior '''
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( nbasis ) * prec

    def set_params( self, params ):
        ''' substitute the parameter into the variable <params>. '''
        self.params = params
        self.sigma2, self.prec = self._trans_params( params )

    def _trans_params( self, params = None ):
        ''' transform the parameter into variance and precision  '''
        if params is None:
            params = self.params

        sigma2 = np.exp( 2 * params )
        prec = 1/sigma2

        return sigma2, prec

class gauss:
    ''' class for gaussian prior '''
    def __init__( self, nbasis, cov = None ):
        self._init_cov( cov )
        self.nbasis = nbasis

    def get_mean( self, params = None ):
        return np.zeros( self.nbasis )

    def get_cov( self, params = None ):
        return self.cov.get_cov( self.nbasis, params )

    def get_prec( self, params = None ):
        return self.cov.get_prec( self.nbasis, params )

    def set_params( self, params ):
        self.cov.set_params( params )

    def _init_cov( self, cov ):
        self.cov = cov
        if cov is None:
            self.cov = cov_const()
