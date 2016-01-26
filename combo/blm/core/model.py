import numpy as np
from .. import inf

class model:
    ''' bayesian linear model '''
    def __init__( self, lik, prior, options = {} ):
        self.prior = prior
        self.lik = lik
        self.nbasis = self.lik.linear.basis.nbasis
        self._init_prior( prior )
        self._set_options( options )
        self.stats = ()

    def comp_stats( self, X, t, Psi = None ):
        if self.method is 'exact':
            inf.exact.comp_stats( blm = self, X = X, t = t, Psi = Psi )
        else:
            pass

    def update_stats( self, x, t, psi = None ):
        if self.method is 'exact':
            self.stats = inf.exact.update_stats( self, x, t, psi )
        else:
            pass

    def get_post_params_mean( self ):
        if self.method is 'exact':
            self.lik.linear.params = inf.exact.get_post_params_mean( blm = self )

    def get_post_fmean( self, X, Psi = None, w = None ):
        if self.method is 'exact':
            fmu = inf.exact.get_post_fmean( self, X, Psi, w )
        else:
            pass
        return fmu

    def sampling( self, w_mu = None, N = 1 ):
        if self.method is 'exact':
            w_hat = inf.exact.sampling( self, w_mu, N )
        else:
            pass
        return w_hat

    def get_post_fcov( self, X, Psi = None, diag = True ):
        if self.method is 'exact':
            fcov = inf.exact.get_post_fcov( self, X, Psi, diag = True)
        else:
            pass
        return fcov

    def _set_options( self, options ):
        self.method = options.get('method','exact')

    def _init_prior( self, prior ):
        ''' initialize the prior distribution '''
        if prior is None:
            prior = prior.gauss( self.nbasis )
        self.prior = prior
