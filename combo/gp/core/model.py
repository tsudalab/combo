import numpy as np
from copy import deepcopy
from ... import opt
from .. import inf
from ... import blm
import learning
import scipy
#import scipy.optimize
from prior import prior

class model:
    def __init__( self, lik, mean, cov, inf = 'exact' ):
        self.lik = lik
        self.prior = prior( mean = mean, cov = cov )
        self.inf = inf
        self.num_params = self.lik.num_params + self.prior.num_params
        self.params = self.cat_params( self.lik.params, self.prior.params )
        self.stats = ()

    def cat_params( self, lik_params, prior_params ):
        ''' concatinate the likelihood and prior parameters '''
        params = np.append( lik_params, prior_params )
        return params

    def decomp_params( self, params = None ):
        if params is None:
            params = np.copy( self.params )

        lik_params = params[0:self.lik.num_params]
        prior_params = params[self.lik.num_params:]
        return lik_params, prior_params

    def set_params( self, params ):
        self.params = params
        lik_params, prior_params =  self.decomp_params( params )
        self.lik.set_params( lik_params )
        self.prior.set_params( prior_params )

    def sub_sampling( self, X, t, N ):
        num_data = X.shape[0]
        if N < num_data:
            index = np.random.permutation( num_data )
            subX = X[index[0:N],:]
            subt = t[index[0:N]]
        else:
            subX = X
            subt = t
        return subX, subt

    def export_blm( self, num_basis ):
        if not hasattr( self.prior.cov, "rand_expans"):
            raise ValueError('The kernel must be.')

        basis_params = self.prior.cov.rand_expans( num_basis )
        basis = blm.basis.fourier( basis_params )
        prior = blm.prior.gauss( num_basis )
        lik = blm.lik.gauss( blm.lik.linear( basis, bias = self.prior.get_mean(1) ), blm.lik.cov(self.lik.params ) )
        blr = blm.model( lik, prior )

        return blr

    def eval_marlik( self, params, X, t, N = None ):
        subX, subt = self.sub_sampling( X, t, N )

        if self.inf is 'exact':
            marlik = inf.exact.eval_marlik( self, subX, subt, params = params )
        else:
            pass

        return marlik

    def get_grad_marlik( self, params, X, t, N = None ):
        subX, subt = self.sub_sampling( X, t, N)

        if self.inf is 'exact':
            grad_marlik = inf.exact.get_grad_marlik( self, subX, subt, params = params )

        return grad_marlik

    def get_params_bound( self ):
        if self.lik.num_params !=0:
            bound = self.lik.get_params_bound()

        if self.prior.mean.num_params !=0:
            bound.extend( self.prior.mean.get_params_bound() )

        if self.prior.cov.num_params !=0:
            bound.extend(self.prior.cov.get_params_bound())
        return bound

    def prepare( self, X, t, params = None ):
        if params is None:
            params = np.copy( self.params )

        if self.inf is 'exact':
            self.stats = inf.exact.prepare( self, X, t, params )
        else:
            pass

    def get_post_fmean( self, X, Z, params = None ):
        if params is None:
            params = np.copy( self.params )

        if self.inf is 'exact':
            post_fmu = inf.exact.get_post_fmean( self, X, Z, params )

        return post_fmu

    def get_post_fcov( self, X, Z, params = None, diag = True):
        if params is None:
            params = np.copy( self.params )

        if self.inf is 'exact':
            post_fcov = inf.exact.get_post_fcov( self, X, Z, params, diag )

        return post_fcov

    def post_sampling( self, X, Z, params = None, N = 1, alpha = 1 ):
        if params is None:
            params = np.copy( self.params )

        fmean = self.get_post_fmean( X, Z, params = None )
        fcov = self.get_post_fcov(X, Z, params = None, diag = False )
        return np.random.multivariate_normal( fmean, fcov * alpha**2, N )

    def predict_sampling( self, X, Z, params = None, N = 1 ):
        if params is None:
            params = np.copy( self.params )

        ndata = Z.shape[0]
        fmean = self.get_post_fmean( X, Z, params = None )
        fcov = self.get_post_fcov( X, Z, params = None, diag = False ) + self.lik.get_cov( ndata )

        return np.random.multivariate_normal( fmean, fcov, N )

    def print_params( self ):
        print ('\n')
        if self.lik.num_params !=0:
            print 'likelihood parameter =  ', self.lik.params

        if self.prior.mean.num_params !=0:
            print 'mean parameter in GP prior: ', self.prior.mean.params

        print 'covariance parameter in GP prior: ', self.prior.cov.params
        print '\n'


    def get_cand_params(self, X, t):
        ''' candidate for parameters '''
        params = np.zeros( self.num_params )
        if self.lik.num_params !=0:
            params[0:self.lik.num_params] = self.lik.get_cand_params(t)

        temp = self.lik.num_params

        if self.prior.mean.num_params!=0:
            params[temp:temp + self.prior.mean.num_params] \
                                = self.prior.mean.get_cand_params(t)

        temp += self.prior.mean.num_params

        if self.prior.cov.num_params!=0:
            params[temp:] = self.prior.cov.get_cand_params( X, t )

        return params

    def fit( self, X, t, config):
        method = config.learning.method

        if method == 'adam':
            adam = learning.adam( self, config)
            params = adam.run( X, t )

        if method in ('bfgs', 'batch' ):
            bfgs = learning.batch( self, config)
            params = bfgs.run( X, t )

        self.set_params( params )
