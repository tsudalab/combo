import numpy as np
import ConfigParser

class set_config:
    def __init__( self, search_config = None, predict_config = None, learning_config = None ):
        if search_config is None:
            search_config = search()
        self.search = search_config

        if predict_config is None:
            predict_config = predict()
        self.predict = predict_config

        if learning_config is None:
            learning_config = adam()
        self.learning = learning_config

    def show( self ):
        self.search.show()
        self.predict.show()
        self.learning.show()

    def load( self, file_name = 'config.ini' ):
        config = ConfigParser.SafeConfigParser()
        config.read( file_name )

        search_config = search()
        self.search = search_config
        self.search.load( config )

        predict_config = predict()
        self.predict = predict_config
        self.predict.load( config )

        temp_dict = config._sections['learning']
        method = temp_dict.get('method','adam')

        if method == 'adam':
            learning_config = adam()
            self.learning = learning_config
            self.learning.load(config)

        if method in ('bfgs', 'batch'):
            learning_config = batch()
            self.learning = learning_config
            self.learning.load(config)

class search:
    def __init__( self ):
        self.dir_name = 'res'
        self.is_disp = True
        self.score = 'TS'
        self.max_search = 100
        self.num_rand_search = 20
        self.alpha = 1.0

    def load( self, config ):
        temp_dict = config._sections['search']
        self.is_disp = temp_dict.get( 'is_disp', True )
        self.dir_name = temp_dict.get( 'dir_name', 'res' )
        self.score = temp_dict.get( 'score', 'TS')
        self.max_search = int( temp_dict.get( 'max_search', 500 ) )
        self.num_rand_search = int( temp_dict.get( 'num_rand_search', 50 ) )
        self.alpha = np.float64( temp_dict.get( 'alpha', 1.0 ) )

    def show( self ):
        print '( search )'
        print 'dir_name: ' + self.dir_name
        print 'score: ' + self.score
        print 'max_search:', self.max_search
        print 'num_rand_search: ', self.num_rand_search
        print 'alpha: ', self.alpha
        print '\n'

class predict:
    def __init__( self ):
        self.is_rand_expans = False
        self.num_basis = 5000

    def load(self, config):
        temp_dict = config._sections['predict']
        self.is_rand_expans = boolean( temp_dict.get('is_rand_expans', False) )
        self.num_basis = int( temp_dict.get('num_basis', 5000) )

    def show( self ):
        print '( predict )'
        print 'is_rand_expans: ', self.is_rand_expans
        print 'num_basis: ', self.num_basis
        print '\n'


class learning( object ):
    def __init__( self ):
        self.is_hyparams_learning = False
        self.is_disp = False
        self.num_disp = 10
        self.num_init_params_search = 20
        self.interval = 20
        self.method = 'adam'

    def show( self ):
        print '( learning )'
        print 'method : ', self.method
        print 'is_hyparams_learning: ', self.is_hyparams_learning
        print 'is_disp: ', self.is_disp
        print 'num_init_params_search: ', self.num_init_params_search
        print 'interval: ', self.interval

    def load( self, config ):
        temp_dict = config._sections['learning']
        self.is_hyparams_learning = boolean( temp_dict.get('is_hyparams_learning', False) )
        self.is_disp = boolean( temp_dict.get('is_disp', False) )
        self.num_disp = int( temp_dict.get('num_disp', 10 ) )
        self.interval = int( temp_dict.get('interval', 20 ) )
        self.num_init_params_search= int( temp_dict.get('num_init_params_search', 20) )
        self.method = temp_dict.get('method','adam')


class batch( learning ):
    def __init__( self ):
        super( batch, self ).__init__()
        self.method = 'bfgs'
        self.max_iter = 200
        self.max_iter_init_params_search = 20
        self.batch_size = 5000

    def show( self):
        super( batch, self ).show()
        print 'max_iter: ', self.max_iter
        print 'max_iter_init_params_search: ', self.max_iter_init_params_search
        print 'batch_size: ', self.batch_size

    def load( self, config ):
        super( batch, self ).load( config )
        temp_dict = config._sections['batch']
        self.max_iter = int( temp_dict.get('max_iter', 200) )
        self.max_iter_init_params_search = int( temp_dict.get('max_iter_init_params_search', 20 ) )
        self.batch_size = int(temp_dict.get('batch_size', 5000))



class online( learning ):
    def __init__( self ):
        super( online, self ).__init__( )
        self.max_epoch = 1000
        self.max_epoch_init_params_search = 50
        self.batch_size = 64
        self.eval_size = 5000

    def show( self ):
        super(online, self).show()
        print 'max_epoch: ', self.max_epoch
        print 'max_epoch_init_params_search: ', self.max_epoch_init_params_search
        print 'batch_size: ', self.batch_size
        print 'eval_size: ', self.eval_size

    def load( self, config ):
        super(online, self).load(config)
        temp_dict = config._sections['online']
        self.max_epoch = int( temp_dict.get('max_epoch', 1000 ) )
        self.max_epoch_init_params_search = int( temp_dict.get('max_epoch_init_params_search', 50 ) )
        self.batch_size = int( temp_dict.get('batch_size', 64 ) )
        self.eval_size = int( temp_dict.get('eval_size', 5000 ) )

class adam( online ):
    def __init__( self ):
        super( adam, self ).__init__( )
        self.method = 'adam'
        self.alpha = 0.001
        self.beta = 0.9
        self.gamma = 0.999
        self.epsilon = 1e-6

    def show( self ):
        super( adam, self ).show()
        print 'alpha = ', self.alpha
        print 'beta = ', self.beta
        print 'gamma = ', self.gamma
        print 'epsilon = ', self.epsilon
        print '\n'

    def load( self, config ):
        super(adam, self).load(config)
        temp_dict = config._sections['adam']
        self.alpha = np.float64( temp_dict.get('alpha', 0.001  ) )
        self.beta = np.float64( temp_dict.get('beta', 0.9 ) )
        self.gamma = np.float64( temp_dict.get('gamma', 0.9999 ) )
        self.epsilon = np.float64( temp_dict.get('epsilon', 1e-6 ) )

def boolean( str ):
    if str == 'True' or str is True:
        return True
    else:
        return False
