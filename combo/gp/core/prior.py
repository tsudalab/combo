import numpy as np
import scipy
from copy import deepcopy

class prior:
    ''' prior of gaussian process '''
    def __init__( self, mean, cov ):
        self.mean = mean
        self.cov = cov
        self.nparams = self.cov.nparams + self.mean.nparams
        self.params = self._concat_params( self.mean.params, self.cov.params )

    def get_mean( self, ndata, params = None ):
        if params is None:
            params = np.copy( self.params )

        return self.mean.get_mean( ndata, params[ 0:self.mean.nparams ] )

    def get_cov( self, X, Z = None, params = None, diag = False ):
        if params is None:
            params = np.copy( self.params )

        return self.cov.get_cov( X, Z, params = params[self.mean.nparams:], diag = diag  )

    def get_grad_mean( self, ndata, params = None ):
        if params is None:
            params = np.copy( self.params )

        mean_params, cov_params = self._decomp_params( params )
        return self.mean.get_grad( ndata, params = mean_params )

    def get_grad_cov( self, X, params = None ):
        if params is None:
            params = np.copy( self.params )
        mean_params, cov_params =self._decomp_params( params )
        return self.cov.get_grad( X, params = cov_params )

    def set_params( self, params ):
        mean_params, cov_params = self._decomp_params( params )
        self.set_mean_params( mean_params )
        self.set_cov_params( cov_params )

    def set_mean_params( self, params ):
        if self.mean.nparams != 0:
            self.params[0:self.mean.nparams ] = params
        self.mean.set_params( params )

    def set_cov_params( self, params ):
        self.params[self.mean.nparams:] = params
        self.cov.set_params( params )

    def sampling( self, X, N = 1 ):
        ''' sampling from GP prior '''
        ndata = X.shape[0]
        G = self.get_cov( X ) + 1e-8 * np.identity( ndata )
        L = scipy.linalg.cholesky( G, check_finite = False )
        Z = np.random.randn( N, ndata )
        return np.dot(Z,L) + self.get_mean( ndata )

    def show( self ):
        print 'instances in mean '
        print ''
        self.mean.show()
        print 'instances in covariance '
        print ''
        self.cov.show()

    def _concat_params( self, mean_params, cov_params ):
        return  np.append( mean_params, cov_params )

    def _decomp_params( self, params = None ):
        if params is None:
            params = np.copy( self.params )

        mean_params = params[0:self.mean.nparams ]
        cov_params = params[self.mean.nparams:]
        return mean_params, cov_params
