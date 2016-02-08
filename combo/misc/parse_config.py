import numpy as np
import ConfigParser

class parse_config:
    def __init__( self, file_name = 'config.ini' ):
        config = ConfigParser.SafeConfigParser()
        config.read( file_name )

        self.search = search( config )
        self.predict = predict( config )

        if self.search.learning_method == 'adam':
            self.learning = adam( config )

        if self.search.learning_method in ('bfgs','batch'):
            self.learning = batch( config )

    def print_config( self ):
        self.search.print_config()
        self.predict.print_config()
        self.learning.print_config()


class search:
    def __init__( self, config ):
        temp_dict = config._sections['search']
        self.is_rand_search = boolean( temp_dict.get('is_rand_search', True) )
        self.is_hyparams_learning = boolean( temp_dict.get('is_hyparams_learning', False) )
        self.score = temp_dict.get( 'score', 'TS')
        self.learning_method = temp_dict.get( 'learning_method', 'adam' )
        self.max_search = int( temp_dict.get( 'max_search', 500 ) )
        self.num_rand_search = int( temp_dict.get( 'num_rand_search', 50 ) )
        self.is_disp = temp_dict.get( 'is_disp', True )
        self.dir_name = temp_dict.get('dir_name','res')
        self.learning_timing = int( temp_dict.get('learning_timing', 25 ) )

    def print_config( self ):
        print '[ config for search ] \n'
        print 'is_rand_search: ', self.is_rand_search
        print 'is_hyparams_learning: ', self.is_hyparams_learning
        print 'score: ' + self.score
        print 'learning_method: ' + self.learning_method
        print 'max_search:', self.max_search
        print 'num_rand_search: ', self.num_rand_search
        print '\n'


class predict:
    def __init__( self, config ):
        temp_dict = config._sections['predict']
        self.is_rand_expans = boolean( temp_dict.get('is_rand_expans', False) )
        self.num_basis = int( temp_dict.get('num_basis', 5000) )

    def print_config( self ):
        print '[ config for prediction ]\n'
        print 'is_rand_expans: ', self.is_rand_expans
        print 'num_basis: ', self.num_basis
        print '\n'


class learning( object ):
    def __init__( self, config ):
        temp_dict = config._sections['learning']
        self.is_disp = boolean( temp_dict.get('is_disp', False) )
        self.num_disp = int( temp_dict.get('num_disp', 10 ) )
        self.is_init_params_search = boolean( temp_dict.get('is_init_params_search', False) )
        self.num_init_params_search= int( temp_dict.get('num_init_params_search', 20) )


    def print_config( self ):
        print '[ config for hyper parameter learning ]'
        print 'is_disp: ', self.is_disp
        print 'num_disp: ', self.num_disp
        print 'is_init_params_search: ', self.is_init_params_search
        print 'num_init_params_search: ', self.num_init_params_search


class batch( learning ):
    def __init__( self, config ):
        super( batch, self ).__init__( config )

        temp_dict = config._sections['batch']
        self.max_iter = int( temp_dict.get('max_iter', 200) )
        self.max_iter_init_params_search = int( temp_dict.get('max_iter_init_params_search', 20 ) )
        self.batch_size = int(temp_dict.get('batch_size', 5000))

    def print_config( self ):
        print '[ config for L-BFGS ] \n'
        print 'max_iter: ', self.max_iter
        print 'max_iter_init_params_search: ', self.max_iter_init_params_search
        print 'batch_size: ', self.batch_size

class online( learning ):
    def __init__( self, config ):
        super( online, self ).__init__( config )

        temp_dict = config._sections['online']
        self.max_epoch = int( temp_dict.get('max_epoch', 1000 ) )
        self.max_epoch_init_params_search = int( temp_dict.get('max_epoch_init_params_search', 50 ) )
        self.batch_size = int( temp_dict.get('batch_size', 64 ) )
        self.eval_size = int( temp_dict.get('eval_size', 5000 ) )

    def print_config( self ):
        super( online, self ).print_config()
        print 'max_epoch: ', self.max_epoch
        print 'max_epoch_init_params_search: ', self.max_epoch_init_params_search
        print 'batch_size: ', self.batch_size
        print 'eval_size: ', self.eval_size
        print '\n'

class adam( online ):
    def __init__( self, config ):
        super( adam, self ).__init__( config )
        temp_dict = config._sections['adam']
        self.alpha = np.float64( temp_dict.get('alpha',0.001) )
        self.beta = np.float64( temp_dict.get('beta',0.9) )
        self.gamma = np.float64( temp_dict.get('gamma',0.999) )
        self.epsilon = np.float64( temp_dict.get('epsilon', 1e-6 ) )

    def print_config( self ):
        super( adam, self ).print_config
        print '[ config for adam ]\n'
        print 'alpha = ', self.alpha
        print 'beta = ', self.beta
        print 'gamma = ', self.gamma
        print 'epsilon = ', self.epsilon
        print '\n'

def boolean( str ):
    if str == 'True' or str is True:
        return True
    else:
        return False
