import numpy as np
import scipy

class prior:
    ''' prior of gaussian process '''
    def __init__( self, mean, cov ):
        self.mean = mean
        self.cov = cov
        self.num_params = self.cov.num_params + self.mean.num_params
        self.params = self.cat_params( self.mean.params, self.cov.params )

    def cat_params( self, mean_params, cov_params ):
        return  np.append( mean_params, cov_params )

    def decomp_params( self, params ):
        if params is None:
            params = np.copy( self.params )

        mean_params = params[0:self.mean.num_params ]
        cov_params = params[self.mean.num_params:]
        return mean_params, cov_params

    def get_mean( self, num_data, params = None ):
        if params is None:
            params = np.copy( self.params )
        return self.mean.get_mean( num_data, params[0:self.mean.num_params] )

    def get_cov( self, X, Z = None, params = None, diag = False ):
        if params is None:
            params = np.copy( self.params )

        return self.cov.get_cov( X, Z, params = params[self.mean.num_params:], diag = diag  )

    def get_grad_mean( self, num_data, params = None ):
        if params is None:
            params = np.copy( self.params )

        mean_params, cov_params = self.decomp_params( params )
        return self.mean.get_grad( num_data, params = mean_params )

    def get_grad_cov( self, X, params = None ):
        if params is None:
            params = np.copy( self.params )
        mean_params, cov_params =self.decomp_params( params )
        return self.cov.get_grad( X, params = cov_params )

    def set_params( self, params ):
        mean_params, cov_params = self.decomp_params( params )
        self.set_mean_params( mean_params )
        self.set_cov_params( cov_params )

    def set_mean_params( self, params ):
        if self.mean.num_params != 0:
            self.params[0:self.mean.num_params ] = params
            self.mean.set_params( params )

    def set_cov_params( self, params ):
        self.params[self.mean.num_params:] = params
        self.cov.set_params( params )

    def sampling( self, X, N = 1 ):
        ''' sampling from GP prior '''
        num_data = X.shape[0]
        G = self.get_cov( X ) + 1e-8 * np.identity( num_data )
        L = scipy.linalg.cholesky( G, check_finite = False )
        Z = np.random.randn( N, num_data )
        return np.dot(Z,L) + self.get_mean( num_data )
