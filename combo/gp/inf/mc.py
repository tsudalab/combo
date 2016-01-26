import numpy as np
import scipy

def eval_marlik( gp, X, t, N, params = None ):
    if params is None:
        params = np.copy(gp.params)

    fhat = gp.prior.sampling( X, N )
        
