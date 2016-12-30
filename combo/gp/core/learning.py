import numpy as np
import scipy.optimize
import six

class batch( object ):
    ''' basis class for batch learning
    '''
    def __init__( self, gp, config ):
        self.gp = gp
        self.config = config

    def run( self, X, t ):
        batch_size = self.config.learning.batch_size
        sub_X, sub_t = self.gp.sub_sampling( X, t, batch_size )

        if self.config.learning.num_init_params_search !=0:
            is_init_params_search = True
        else:
            is_init_params_search = False


        if is_init_params_search:
            params = self.init_params_search( sub_X, sub_t )
        else:
            params = np.copy( self.gp.params )

        params = self.one_run( params, sub_X, sub_t )
        return params

    def one_run( self, params, X, t, max_iter = None ):
        is_disp = True

        if max_iter is None:
            is_disp = self.config.learning.is_disp
            max_iter = int(self.config.learning.max_iter)

        args = (X, t)
        bound = self.gp.get_params_bound()
        res = scipy.optimize.minimize( fun=self.gp.eval_marlik, args= args, \
                 x0=params, method='L-BFGS-B', jac=self.gp.get_grad_marlik, \
                 bounds = bound, options={'disp':is_disp,'maxiter':max_iter})

        return res.x

    def init_params_search( self, X, t ):
        num_init_params_search = self.config.learning.num_init_params_search
        max_iter = int(self.config.learning.max_iter_init_params_search)
        min_params = np.zeros( self.gp.num_params )
        min_marlik = np.inf

        for i in range(num_init_params_search):
            params = self.gp.get_cand_params( X, t )
            params = self.one_run( params, X, t, max_iter )
            marlik = self.gp.eval_marlik( params, X, t )

            if min_marlik > marlik:
                min_marlik = marlik
                min_params = params

        #print('minimum marginal likelihood = ', min_marlik)
        return min_params


class online( object ):
    ''' base class for online learning '''
    def __init__( self, gp, config ):
        self.gp = gp
        self.config = config
        self.num_iter = 0

    def run( self, X, t ):
        if self.config.learning.num_init_params_search != 0:
            is_init_params_search = True
        else:
            is_init_params_search = False


        if is_init_params_search:
            print('Start the initial hyper parameter searching ...')
            params = self.init_params_search( X, t )
            print('Done\n')
        else:
            params = np.copy( self.params )

        print('Start the hyper parameter learning ...')
        params = self.one_run( params, X, t )
        print('Done\n')

        return params

    def one_run( self, params, X, t, max_epoch = None ):
        num_data = X.shape[0]
        is_disp = False
        batch_size = self.config.learning.batch_size

        if batch_size > num_data:
            batch_size = num_data

        if max_epoch is None:
            max_epoch = self.config.learning.max_epoch
            is_disp = self.config.learning.is_disp

        num_disp = self.config.learning.num_disp
        eval_size = self.config.learning.eval_size
        eval_X, eval_t = self.gp.sub_sampling( X, t, eval_size )
        timing = range( 0, max_epoch, int( np.floor( max_epoch / num_disp ) ) )
        temp = 0

        for num_epoch in range( 0, max_epoch ):
            perm = np.random.permutation( num_data )

            if is_disp and temp < num_disp and num_epoch == timing[temp]:
                self.disp_marlik( params, eval_X, eval_t, num_epoch )
                temp += 1

            for n in six.moves.range( 0, num_data, batch_size ):
                tmp_index = perm[n:n + batch_size]
                if len(tmp_index) == batch_size:
                    self.num_iter += 1
                    subX = X[tmp_index,:]
                    subt = t[tmp_index]
                    params += self.get_one_update( params, subX, subt )

        if is_disp:
            self.disp_marlik( params, eval_X, eval_t, num_epoch + 1 )

        self.reset()
        return params


    def disp_marlik( self, params, eval_X, eval_t, num_epoch = None ):
        marlik = self.gp.eval_marlik( params, eval_X, eval_t )
        if num_epoch is not None:
            print(num_epoch, end="")
            print('-th epoch', end="")

        print('marginal likelihood', marlik)


    def init_params_search( self, X, t ):
        ''' initial parameter searchs '''
        num_init_params_search = self.config.learning.num_init_params_search
        is_disp = self.config.learning.is_disp
        max_epoch = self.config.learning.max_epoch_init_params_search
        eval_size = self.config.learning.eval_size
        eval_X, eval_t = self.gp.sub_sampling( X, t, eval_size )
        min_params = np.zeros( self.gp.num_params )
        min_marlik = np.inf

        for i in range(num_init_params_search):
            params = self.gp.get_cand_params( X, t )

            params = self.one_run( params, X, t, max_epoch )
            marlik = self.gp.eval_marlik( params, eval_X, eval_t )

            if min_marlik > marlik:
                min_marlik = marlik
                min_params = params

        #print('minimum marginal likelihood = ', min_marlik)
        return min_params

    def get_one_update( self, params, X, t ):
        raise NotImplementedError

class adam( online ):
    ''' default '''
    def __init__( self, gp, config ):
        super(adam, self).__init__( gp, config )

        self.alpha = self.config.learning.alpha
        self.beta = self.config.learning.beta
        self.gamma = self.config.learning.gamma
        self.epsilon = self.config.learning.epsilon
        self.m = np.zeros( self.gp.num_params )
        self.v = np.zeros( self.gp.num_params )

    def reset( self ):
        self.m = np.zeros( self.gp.num_params )
        self.v = np.zeros( self.gp.num_params )
        self.num_iter = 0

    def get_one_update( self, params, X, t ):
        grad = self.gp.get_grad_marlik( params, X, t )
        self.m = self.m * self.beta + grad * ( 1 - self.beta )
        self.v = self.v * self.gamma + grad**2 * ( 1 - self.gamma )
        hat_m = self.m / ( 1 - self.beta ** ( self.num_iter  ) )
        hat_v = self.v / ( 1 - self.gamma ** ( self.num_iter ) )
        return - self.alpha * hat_m / ( np.sqrt( hat_v ) + self.epsilon )
