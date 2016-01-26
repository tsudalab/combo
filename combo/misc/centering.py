import numpy as np

def centering( X ):
    stdX  = np.std( X, 0 )
    index = np.where( stdX !=0 )
    return ( X[:,index[0]] - np.mean( X[:,index[0]], 0 ) ) / stdX[ index[0] ]
