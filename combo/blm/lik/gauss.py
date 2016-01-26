import numpy as numpy

class gauss:
    def __init__( self, linear, cov ):
        self.linear = linear
        self.cov = cov        
        self.stats = ()

    def get_cov( self, N, params = None ):
        if params is None:
            params = np.copy( self.cov.params )

        return self.cov.get_cov( N, params )

    def get_prec( self, N, params = None ):
        if params is None:
            params = np.copy( self.cov.params )

        return self.cov.get_cov( N, params )

    def get_basis( self, X ):
        return self.linear.basis.get_basis( X )

    def get_mean( self, X, Psi = None, params = None, bias = None ):
        return self.linear.get_mean( X, Psi, params, bias )

    def set_params( self, params ):
        self.linear.set_params( params )

    def set_bias( self, bias ):
        self.linear.set_bias( bias )
