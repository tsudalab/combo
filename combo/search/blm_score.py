import numpy as np

def TS( blm, Xtest, Psi = None ):
    if Psi is None:
        Psi = blm.lik.get_basis( Xtest )

    w_hat = blm.sampling()
    score = Psi.dot( w_hat ) + blm.lik.linear.bias

    return score
