# -*- coding:utf-8 -*-
import numpy as np
from scipy import spatial
import _src
from _src.enhance_gauss import grad_width64

max_params = np.log(1e5)
min_params = np.log(1e-5)

class gauss:
    ''' Gaussian kernel '''
    def __init__( self, ndims, params = None, ard = False ):
        ''' initilization '''
        self.ndims = ndims
        self._ard = ard
        self._init_params( params )
        self.nparams = len( self.params )
        self.params = self._supp_params()
        tmp, self.width, self.scale2 = self._trans_params()

    def set_params( self, params ):
        self._check_len_params( params )
        params = self._supp_params( params )
        self.params = params
        tmp, self.width, self.scale2 = self._trans_params( params )

    def get_cov( self, X, Z = None, params = None, diag = False ):
        ''' compute the covariance ( gram ) matrix '''
        self._check_len_input_dims( X )
        params, width, scale2 = self._trans_params( params )

        if Z is None:
            if diag:
                G = scale2 * np.ones(X.shape[0])
            else:
                pairwise_dists = spatial.distance.squareform( \
                                    spatial.distance.pdist( X/width, 'euclidean' )**2 )
                G = np.exp(- 0.5 * pairwise_dists ) * scale2
        else:
            self._check_len_input_dims( Z )
            pairwise_dists = spatial.distance.cdist( X/width, Z/width, 'euclidean')**2
            G = np.exp(- 0.5 * pairwise_dists ) * scale2

        return G

    def get_grad( self, X, params = None ):
        ''' compute the gradient with respect to kernel parameters '''
        ndata = X.shape[0]
        self._check_len_input_dims(X)
        params, width, scale2 = self._trans_params( params )

        # gram matrix
        G = self.get_cov( X, params = params )

        gradG = np.zeros((self.nparams, ndata, ndata))
        if self._ard:
            gradG[0:self.nparams-1,:,:] = grad_width64(X, width, G)
        else:
            pairwise_dists = spatial.distance.pdist( X/width,'euclidean' )
            gradG[0,:,:] = G * spatial.distance.squareform( pairwise_dists**2 )

        gradG[-1,:,:] = 2 * G
        return gradG

    def rand_expans( self, nbasis = 5000, params = None ):
        params, width, scale2 = self._trans_params(params)
        amp = np.sqrt( ( 2 * scale2 )/nbasis )
        W = np.random.randn( nbasis, self.ndims )/width
        b = np.random.rand( nbasis ) * 2 * np.pi
        return (W, b, amp)

    def show( self ):
        print ' '
        print ' ndims   = ', self.ndims
        print ' _ard    = ', self._ard
        print ' nparams = ', self.nparams
        print ' params  = ', self.params
        print ' width   = ', self.width
        print ' scale2  = ', self.scale2

    def _trans_params( self, params = None ):
        ''' decompose the parameters into two parts, width and scale parameters '''
        if params is None:
            params = np.copy( self.params )
        else:
            self._check_len_params( params )

        width = np.exp( params[0:-1])
        scale2 = np.exp( 2 * params[-1] )

        return params, width, scale2

    def _init_params( self, params ):
        self.params = params
        if params is None:
            ''' when <params> is not defined '''
            if self._ard:
                self.params = np.ones( self.ndims ) * np.log( 1 )
                self.params = np.append( self.params, np.log(np.sqrt( 1 )))
            else:
                self.params = np.log(1)
                self.params = np.append(self.params, np.log(np.sqrt(1)))
        else:
            ''' when <params> is defined '''
            if self._ard:
                if len(params) == self.ndims + 1:
                    pass
                elif len(params) == 2:
                    self.params = np.ones( self.ndims ) * params[0]
                    self.params = np.append( self.params, params[1] )
                else:
                    message = "The length of <params> must be same as %d or 2." %( self.ndims + 1 )
                    raise ValueError(message)
            elif len( params ) != 2:
                raise ValueError("The length of <params> must be same as 2.")
            else:
                pass

    def _check_len_params( self, params ):
        ''' check the length of params '''
        if len( params ) != self.nparams:
            message = 'The length of < params > must be same as %d.' %(self.nparams)
            raise ValueError(message)

    def _check_len_input_dims( self, X ):
        ''' check the length of 1-axis of input matrix '''
        if self.ndims != X.shape[1]:
            message = "The length of 1-axis of input matrix < X > must be same as %d." %(self.ndims)
            raise ValueError(message)

    def _supp_params( self, params = None ):
        ''' suppress the absolute of parameters '''
        if params is None:
            params = np.copy( self.params )

        index = np.where( params > max_params)
        params[index[0]] = max_params
        index = np.where( params < min_params )
        params[index[0]] = min_params

        return params
