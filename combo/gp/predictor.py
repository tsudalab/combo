import numpy as np
import cov
import lik
import mean
import core
from ..predictor import base_predictor

class predictor( base_predictor ):
    ''' predictor '''
    def __init__( self, config, model = None ):
        super( predictor, self ).__init__( config, model )

    def fit(self, training, num_basis=None):
        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        self.model.fit(training.X, training.t, self.config)
        self.delete_stats()

    def get_basis( self, *args, **kwds ):
        pass

    def get_post_params( self, *args, **kwds ):
        pass

    def prepare( self, training ):
        self.model.prepare( training.X, training.t )

    def delete_stats( self ):
        self.model.stats = None

    def get_post_fmean( self, training, test ):
        if self.model.stats is None:
            self.prepare( training )
        return self.model.get_post_fmean( training.X, test.X )

    def get_post_fcov( self, training, test, diag = True ):
        if self.model.stats is None:
            self.prepare(training)
        return self.model.get_post_fcov( training.X, test.X, diag = diag )

    def get_post_samples( self, training, test, alpha = 1 ):
        if self.model.stats is None:
            self.prepare( training )
        return self.model.post_sampling( training.X, test.X, alpha = alpha )

    def get_predict_samples( self, training, test, N = 1  ):
        if self.model.stats is None:
            self.prepare( training )
        return self.model.predict_sampling( training.X, test.X, N = N )
