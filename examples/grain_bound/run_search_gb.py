import numpy as np
import argparse
import combo

def load():
    A = np.asarray(np.loadtxt('s5-210.csv',skiprows=1,delimiter=','))
    X = A[:,0:3]
    t = A[:,3]
    return X, t

class simulator:
    def __init__( self ):
        X, t = load()

    def __call__( self, action ):
        return -t[ action ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description = 'This script is to evaluate the performance of COMBO' )
    parser.add_argument('--seed', default= 0, type=int, help='seed parameter')
    parser.add_argument('--score', default= 'TS', type=str, help='score function')
    parser.add_argument('--dir_name', default= 'res', type=str, help='directory name')
    parser.add_argument('--num_basis', default= 0, type=int, help='number of basis functions')
    args = parser.parse_args()

    X, t = load()
    X = combo.misc.centering( X )
    
    mean = combo.gp.mean.const()
    cov = combo.gp.cov.gauss( X.shape[1], ard = False )
    lik = combo.gp.lik.gauss()
    gp = combo.gp.model( mean = mean, cov = cov, lik = lik )

    config = combo.misc.set_config()
    config.load('config.ini')

    config.search.score = args.score
    config.search.dir_name = args.dir_name

    if args.num_basis == 0:
        config.predict.is_rand_expans = False
    else:
        config.predict.is_rand_expans = True
        config.predict.num_basis = args.num_basis

    config.show()

    simu = simulator()
    search = combo.search.bayes_policy(simu, X, config)
    search.set_seed(args.seed)
    search.run( gp )
