# -*- coding:utf-8 -*-
import numpy as np
import scipy
#from scipy.stats import multivariate_normal

class gauss:
    ''' Gaussian likelihood function '''
    def __init__( self, std = 1, max_params = 1e6, min_params = 1e-6 ):
        self.min_params = np.log( min_params )
        self.max_params = np.log( max_params )
        self.num_params = 1
        self.std = std
        self.params = np.log( std )
        self.set_params( self.params )

    def supp_params( self, params = None ):
        if params is None:
            params = np.copy( params )

        if params > self.max_params :
            params = self.max_params

        if params < self.min_params :
            params =  self.min_params

        return params

    def trans_params( self, params = None ):
        if params is None:
            params = np.copy( self.params )

        std = np.exp( params )
        return std

    def get_params_bound( self ):
        bound = [ ( self.min_params, self.max_params ) for i in range(0, self.num_params) ]
        return bound

    def get_cov( self, num_data, params = None ):
        std = self.trans_params( params )
        var = std ** 2
        return var * np.identity( num_data )

    def get_grad( self, num_data, params = None ):
        std = self.trans_params( params )
        var = std ** 2
        return var * np.identity( num_data ) * 2

    def set_params( self, params ):
        self.params = self.supp_params( params )
        self.std = self.trans_params( params )

    def get_cand_params( self, t ):
        return np.log( np.std(t) / 10 )
