import numpy as np
import time
from res import res
import blm_score

class bayes:
    def __init__( self, call_sim, search_options = {}, learn_options ={} ):
        self.call_sim = call_sim
        self.options = learn_options
        self._load_options( search_options )
        self.res = res( self.max_iter, self.directory, ( search_options, learn_options ) )
        self.seed = None

    def set_seed( self, seed ):
        self.seed = seed
        np.random.seed( int( self.seed ) )

    def run( self, gp, Xtest,  Xtrain = None, ttrain = None,  process = None ):
        self.candidates = xrange( 0, Xtest.shape[0] )

        if Xtrain is None and ttrain is None:
            Xtrain = np.empty([0, gp.prior.cov.ndims ])
            ttrain = np.empty(0)

        self.res.set_ttrain( ttrain )

        if self.nburn_in != 0:
            print 'start random sampling ...'
            Xtrain, Xtest = self.rand_search( Xtest, Xtrain, ttrain )

        if self.ker_expans and self.nburn_in != self.max_iter :
            print 'start bayes search with kernel expansions ... '
            self.blm_search( gp, Xtest, Xtrain )

        self.save( process )

    def save( self, process ):
        if process is None and self.seed is None:
            process = 0

        if process is None:
            process = self.seed

        self.res.save( process )

    def rand_search( self, Xtest, Xtrain, ttrain ):
        ''' random sampling '''
        for n in xrange( self.nburn_in ):
            tmp = np.random.random_integers( 0, len( self.candidates ) -1 )
            query = self.candidates[ tmp ]
            t = self.call_sim( query  )
            self.res.write( query, t )
            x = Xtest[ tmp, : ]
            Xtrain = np.vstack( ( Xtrain, x ) )
            Xtest = np.delete( Xtest, tmp, 0 )
            self.candidates = np.delete( self.candidates , tmp )
            self.display(n)
            self.res.query_time[n] = 0
        return Xtrain, Xtest

    def blm_search( self, gp, Xtest, Xtrain ):
        nlearn = 0
        if self.learn:
            if self.reset_init_params is not None:
                self.reset_init_params( gp, Xtrain, self.res.ttrain )
            gp.learn( Xtrain, self.res.ttrain, options = self.options  )
            nlearn += 1

        blm = gp.export_blm( self.nbasis )
        Psi_train = blm.lik.get_basis( Xtrain )
        Psi_test  = blm.lik.get_basis( Xtest )
        blm.comp_stats( Xtrain, self.res.ttrain, Psi = Psi_train )

        for n in xrange( self.nburn_in, self.max_iter ):
            st = time.time()
            if self.score == 'TS':
                score = blm_score.TS( blm, Xtest, Psi_test  )
            tmp = np.argmax( score )
            query = self.candidates[ tmp ]

            x = Xtest[ tmp,:]
            psi = Psi_test[ tmp,: ]

            t = self.call_sim( query )

            self.res.write( query, t )
            Xtrain = np.vstack(( Xtrain, x ) )
            Psi_train = np.vstack( ( Psi_train, psi ) )

            Xtest = np.delete( Xtest, tmp, 0 )
            Psi_test = np.delete( Psi_test, tmp, 0 )
            self.candidates = np.delete( self.candidates, tmp )
            self.display(n)

            if self.learn and isinstance(self.timing, list)  \
            and nlearn < len( self.timing ) and n == self.timing[ nlearn ]:

                if self.reset_init_params is not None:
                    self.reset_init_params( gp, Xtrain, self.res.ttrain )

                gp.learn( Xtrain, self.res.ttrain, options = self.options )
                blm = gp.export_blm( self.nbasis )
                Psi_train = blm.lik.get_basis( Xtrain )
                Psi_test = blm.lik.get_basis( Xtest )
                blm.comp_stats( Xtrain, self.res.ttrain, Psi = Psi_train )
                nlearn += 1
            else:
                blm.update_stats( x, t, psi)

            self.res.query_time[n] = time.time()-st


    def display( self, n ):
        fn  = "{0:05d}".format(n+1)
        fx  = "{0:5.4f}".format(self.res.ttrain[n])
        fbest = "{0:5.4f}".format(self.res.best_ttrain[n])
        print fn, '-th step:  f(x) = ', fx,  'fbest = ', fbest


    def _load_options( self, options ):
        self.nburn_in = options.get( 'nburn_in', 50 )
        self.learn = options.get('learn', True )
        self.timing = options.get('learn_timing', 25)
        self.ker_expans = options.get('ker_expans', True)
        self.nbasis = options.get('nbasis', 5000)
        self.score = options.get('score', 'TS')
        self.max_iter = options.get('max_iter', 500)
        self.directory = options.get('directory_name', 'res')
        self.disp = options.get('disp', True)
        self.reset_init_params = options.get('reset_init_params', None )

        if self.max_iter < self.nburn_in:
            raise ValueError('< nburn_in > must be small than < max_iter >.')

        if not isinstance( self.timing, list):
            if self.timing ==0:
                pass
            else:
                self.timing = range(self.nburn_in, self.max_iter, self.timing )
