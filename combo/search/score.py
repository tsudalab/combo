import numpy as np
import scipy.stats

def EI( predictor, training, test, fmax = None ):
    fmean = predictor.get_post_fmean( training, test )
    fcov = predictor.get_post_fcov( training, test )
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max( predictor.get_post_fmean( training, training ))

    temp1 = ( fmean - fmax )
    temp2 = temp1 / fstd
    score = temp1 * scipy.stats.norm.cdf( temp2 ) + fstd * scipy.stats.norm.pdf( temp2 )
    return score


def PI( predictor, training, test, fmax = None ):
    fmean = predictor.get_post_fmean( training, test )
    fcov = predictor.get_post_fcov( training, test )
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max( predictor.get_post_fmean( training, training ) )

    temp = ( fmean - fmax )/fstd
    score = scipy.stats.norm.cdf( temp )
    return score

def TS( predictor, training, test, alpha = 1 ):
    score = predictor.get_post_samples( training, test, alpha = alpha )

    try:
        score.shape[1]
        score[0,:]
    except:
        pass

    return score
