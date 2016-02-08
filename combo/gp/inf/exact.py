import numpy as np
import scipy
from ... import misc
from copy import deepcopy

def eval_marlik( gp, X, t, params = None ):
    ndata, ndims = X.shape
    lik_params, prior_params = gp.decomp_params( params )

    fmu = gp.prior.get_mean( ndata, params = prior_params )
    G = gp.prior.get_cov( X, params = prior_params )
    B = gp.lik.get_cov( ndata, params = lik_params )

    A = G + B + 1e-8 * np.identity( ndata )
    res = t - fmu
    L = scipy.linalg.cholesky( A, check_finite = False )
    alpha = scipy.linalg.solve_triangular( L.transpose(), res, \
                            lower=True, overwrite_b = False, check_finite = False)
    marlik = 0.5 * ndata * np.log( 2 * np.pi ) + np.sum(np.log(np.diag(L)))  \
                                             + 0.5 * np.inner( alpha, alpha )
    return marlik

def get_grad_marlik( gp, X, t, params = None ):
    ndata, ndims  = X.shape
    lik_params, prior_params = gp.decomp_params( params )

    fmu = gp.prior.get_mean( ndata, prior_params )
    G = gp.prior.get_cov( X, params = prior_params )
    B = gp.lik.get_cov( ndata, lik_params )

    A = G + B + 1e-8 * np.identity( ndata )
    L = scipy.linalg.cholesky( A, check_finite = False )
    res = t - fmu
    alpha = misc.gauss_elim( L, res )
    invA = scipy.linalg.inv( A, check_finite = False)

    grad_marlik = np.zeros( gp.num_params )

    ''' lik '''
    if gp.lik.num_params !=0:
        lik_grad = gp.lik.get_grad( ndata, lik_params )
        temp = lik_grad.dot( alpha )
        grad_marlik[0:gp.lik.num_params] = -0.5 * temp.dot(alpha) + 0.5 * misc.traceAB2( invA, lik_grad )

    ntemp = gp.lik.num_params
    ''' prior '''
    if gp.prior.mean.num_params !=0:
        mean_grad = gp.prior.get_grad_mean( ndata, prior_params )
        grad_marlik[ntemp:ntemp+gp.prior.mean.num_params] = - np.inner(alpha, mean_grad)

    ntemp += gp.prior.mean.num_params

    if gp.prior.cov.num_params != 0:
        cov_grad = gp.prior.get_grad_cov( X, prior_params )
        temp = cov_grad.dot(alpha)
        grad_marlik[ntemp:] = -0.5 * temp.dot(alpha) + 0.5 * misc.traceAB3( invA, cov_grad )

    return grad_marlik

def prepare( gp, X, t, params = None ):
    ndata = X.shape[0]
    ndims = X.shape[1]

    if params is None:
        params = np.copy( gp.params )

    lik_params, prior_params = gp.decomp_params( params )

    G = gp.prior.get_cov( X, params = prior_params )
    fmu = gp.prior.get_mean(ndata, params = prior_params )
    B = gp.lik.get_cov(ndata, params = lik_params )
    A = G + B + 1e-8 * np.identity(ndata)
    L = scipy.linalg.cholesky(A, check_finite = False)
    residual = t - fmu
    alpha = misc.gauss_elim( L, residual )
    stats = (L, alpha)

    return stats

def get_post_fmean( gp, X, Z, params = None ):
    ''' calculate the mean of posterior  '''

    ndata = X.shape[0]
    ndims = X.shape[1]
    ntest = Z.shape[0]

    lik_params, prior_params = gp.decomp_params( params )

    alpha = gp.stats[1]

    fmu = gp.prior.get_mean(ntest)
    G = gp.prior.get_cov( X=Z, Z=X, params = prior_params )

    return G.dot(alpha) + fmu

def get_post_fcov(gp, X, Z, params = None, diag = True ):
    ''' calculate the covariance of posterior  '''
    lik_params, prior_params = gp.decomp_params( params )

    L = gp.stats[0]
    alpha = gp.stats[1]

    G = gp.prior.get_cov( X=X, Z=Z, params = prior_params )

    invLG = scipy.linalg.solve_triangular( L.transpose(), G, lower=True,
                                      overwrite_b = False, check_finite=False )

    if diag:
        diagK = gp.prior.get_cov(X=Z, params = prior_params, diag=True )
        diag_invLG2 = misc.diagAB( invLG.transpose(), invLG )
        post_cov = diagK - diag_invLG2
    else:
        K = gp.prior.get_cov( X = Z, params = prior_params )
        post_cov = K - np.dot( invLG.transpose(), invLG )

    return post_cov
