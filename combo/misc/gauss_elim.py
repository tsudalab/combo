import numpy as np
import scipy

def gauss_elim( L, t ):
    alpha = scipy.linalg.solve_triangular( L.transpose(), t, \
                        lower=True, overwrite_b = False, check_finite=False )

    alpha = scipy.linalg.solve_triangular( L, alpha, \
                        lower=False, overwrite_b = False, check_finite=False )
    return alpha
