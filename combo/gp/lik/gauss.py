# -*- coding:utf-8 -*-
import numpy as np
import scipy
from scipy.stats import multivariate_normal

max_params = np.log(1e5)
min_params = np.log(1e-5)

class gauss:
    ''' Gaussian likelihood function '''
    def __init__( self, params = None ):
        self._init_params( params )
        self._params = self._supp_params(self.params)
        self.set_params( params )
        self.nparams = 1

    def get_cov( self, ndata, params = None ):
        params, sigma2 = self._trans_params( params )
        return sigma2 * np.identity( ndata )

    def get_grad( self, ndata, params = None ):
        params, sigma2 = self._trans_params( params )
        return np.identity( ndata ) * sigma2 * 2

    def set_params( self, params ):
        params = self._supp_params( params )
        self.params, self.sigma2 = self._trans_params( params )

    def get_lnpdf( self, t, y,  params = None ):
        params, sigma2 = self._trans_params(params)
        ndata = t.shape[0]
        lnpdf = multivariate_normal.logpdf( y, t, self.get_cov(ndata) )
        return -0.5 * lnpdf

    def _trans_params(self, params = None):
        ''' transform the kernel parameters '''
        if params is None:
            params = np.copy(self.params)

        sigma2 = np.exp( 2 * params )
        return params, sigma2

    def _supp_params( self, params = None ):
        ''' suppress the absolute of parameters  '''
        params, scale2 = self._trans_params( params )
        if params > max_params:
            params = max_params
        if params < min_params:
            params = min_params
        return params

    def _init_params( self, params ):
        if params is None:
            self.params = np.log(1)
        else:
            self.params = params
