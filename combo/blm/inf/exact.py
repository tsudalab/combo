import numpy as np
import scipy
from ... import misc

def prepare( blm, X, t, Psi = None ):
    if Psi is None:
        Psi= blm.lik.get_basis( X )
    PsiT = Psi.transpose()
    G = np.dot(PsiT, Psi) * blm.lik.cov.prec
    A = G + blm.prior.get_prec()
    L = scipy.linalg.cholesky( A, check_finite = False )
    b = PsiT.dot( t - blm.lik.linear.bias )
    alpha = misc.gauss_elim( L, b)
    blm.stats = (L,b,alpha)

def update_stats( blm, x, t, psi = None ):
    ''' update efficient statistics '''
    if psi is None:
        psi = blm.lik.get_basis( x )
    L = blm.stats[0]
    b = blm.stats[1] + (t - blm.lik.linear.bias )* psi
    misc.cholupdate( L, psi * np.sqrt( blm.lik.cov.prec ) )
    alpha = misc.gauss_elim( L, b )
    return ( L, b, alpha )

def sampling( blm, w_mu = None, N=1 ):
    if w_mu is None:
        w_mu = get_post_params_mean( blm )
    if N==1:
        z = np.random.randn( blm.nbasis )
    else:
        z = np.random.randn( blm.nbasis, N )

    L = blm.stats[0]
    invLz = scipy.linalg.solve_triangular( L, z, \
                    lower=False, overwrite_b = False, check_finite = False )
    return (invLz.transpose() + w_mu).transpose()

def get_post_params_mean( blm ):
    return blm.stats[2] * blm.lik.cov.prec

def get_post_fmean( blm, X, Psi = None, w = None ):
    if Psi is None:
        Psi = blm.lik.linear.basis.get_basis( X )

    if w is None:
        w = get_post_params_mean( blm )
    return Psi.dot(w) + blm.lik.linear.bias

def get_post_fcov( blm, X, Psi = None, diag = True ):
    if Psi is None:
        Psi = blm.lik.linear.basis.get_basis( X )

    L = blm.stats[0]
    R = scipy.linalg.solve_triangular(L.transpose(), Psi.transpose(), \
                    lower=True, overwrite_b = False, check_finite=False )
    RT = R.transpose()

    if diag is True:
        fcov = misc.diagAB( RT, R )
    else:
        fcov = np.dot( RT, R )

    return fcov
