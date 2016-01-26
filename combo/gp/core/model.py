import numpy as np
from copy import deepcopy
from ... import opt
from .. import inf
from ... import blm
import scipy.optimize
from prior import prior

class model:
    def __init__( self, lik, mean, cov, hyprior = None, options ={} ):
        self.lik = lik
        self.prior = prior( mean = mean, cov = cov )
        self.hyprior = hyprior
        self.options = self.set_options( options )
        self.nparams = self.lik.nparams + self.prior.nparams
        self.params = self._concat_params( self.lik.params, self.prior.params )
        self.stats = ()

    def eval_marlik( self, params, X, t, N = None ):
        subX, subt = self._sub_sampling( X, t, N )

        if self.mode is 'exact':
            marlik = inf.exact.eval_marlik( self, subX, subt, params = params )
        else:
            pass

        return marlik

    def get_grad_marlik( self, params, X, t, N = None ):
        subX, subt = self._sub_sampling( X, t, N)

        if self.mode is 'exact':
            grad_marlik = inf.exact.get_grad_marlik( self, subX, subt, params = params )

        return grad_marlik

    def learn( self, X, t, params = None, options = {} ):
        if params is None:
            params = np.copy( self.params )

        method = options.get('method','adam')


        if method == 'adam':
            subN = int( options.get( 'subN_learn', 64  ) )
            disp = options.get( 'disp', False )
            interval_disp = options.get( 'interval_disp', 1000 )
            max_epoch = options.get('max_epoch', 10000)
            subN_eval = int( options.get('subN_eval', 1024) )

            args = ( X, t, subN )
            adam = opt.adam( params, self.get_grad_marlik, options = options )

            if disp is False:
                adam.run( *args )
                params = adam.params
            else:
                subX, subt = self._sub_sampling( X, t, subN_eval )
                for epoch in xrange( max_epoch ):
                    if np.mod(epoch, interval_disp) == 0:
                        message = 'number of epochs = %05d, negative marginal likelihood = %05.4f' \
                                    %( epoch, self.eval_marlik( params, subX, subt ) )
                        print message
                    update = adam.update( params, *args )
                    params += update

                message = 'number of epochs = %05d, negative marginal likelihood = %05.4f \n ' \
                                         %( epoch+1, self.eval_marlik( params, subX, subt ) )
                print message

        elif method == 'BFGS':
            ndata = X.shape[0]
            subN = int(options.get('subN_learn', ndata))
            disp = options.get( 'disp', False )
            subN = min(subN, ndata)
            if subN != ndata:
                subX, subt = self._sub_sampling( X, t, subN )
            else:
                subX = X
                subt = t

            args = ( subX, subt )
            bounds = [( np.log(1e-5) , np.log(1e5) )  for i in range(0, self.nparams)]
            res = scipy.optimize.minimize(fun = self.eval_marlik,
                             x0=self.params, args=args,  bounds = bounds, method='L-BFGS-B',
                              jac = self.get_grad_marlik, options={'gtol': 1e-4, 'disp': disp})
            params = res.x

        else:
            pass

        self.set_params( params )
        self.show_params()
        self.stats = ()

    def comp_stats( self, X, t, params = None ):
        if params is None:
            params = np.copy( self.params )

        if self.mode is 'exact':
            self.stats = inf.exact.comp_stats( self, X, t, params )
        else:
            pass

    def get_post_fmean( self, X, Z, params = None ):
        if params is None:
            params = np.copy( self.params )

        if self.mode is 'exact':
            post_fmu = inf.exact.get_post_fmean( self, X, Z, params )

        return post_fmu

    def get_post_fcov( self, X, Z, params = None, diag = True):
        if params is None:
            params = np.copy( self.params )

        if self.mode is 'exact':
            post_fcov = inf.exact.get_post_fcov( self, X, Z, params, diag )

        return post_fcov

    def show_params( self ):
        if self.lik.nparams !=0:
            print 'likelihood parameter: ', self.lik.params

        if self.prior.mean.nparams !=0:
            print 'mean parameter of prior: ', self.prior.mean.params

        print 'covariance parameter of prior: ', self.prior.cov.params
        print '\n'

    def set_params( self, params ):
        self.params = params
        lik_params, prior_params =  self._decomp_params( params )
        self.lik.set_params(lik_params)
        self.prior.set_params(prior_params)

    def set_options( self, options ):
        self.mode = options.get('mode','exact')

    def export_blm( self, nbasis ):
        if not hasattr( self.prior.cov, "rand_expans"):
            raise ValueError('The kernel must be.')

        basis_params = self.prior.cov.rand_expans( nbasis )
        basis = blm.basis.fourier( basis_params )
        prior = blm.prior.gauss( nbasis )
        lik = blm.lik.gauss( blm.lik.linear( basis, bias = self.prior.get_mean(1) ), blm.lik.cov(self.lik.params ) )
        blr = blm.model( lik, prior )

        return blr

    def _sub_sampling( self, X, t, N ):
        ndata = X.shape[0]
        if N < ndata:
            index = np.random.permutation( ndata )
            subX = X[index[0:N],:]
            subt = t[index[0:N]]
        else:
            subX = X
            subt = t
        return subX, subt

    def _concat_params( self, lik_params, prior_params ):
        params = np.append( lik_params, prior_params )
        return params

    def _decomp_params( self, params = None ):
        if params is None:
            params = np.copy( self.params )

        lik_params = params[0:self.lik.nparams]
        prior_params = params[self.lik.nparams:]
        return lik_params, prior_params
