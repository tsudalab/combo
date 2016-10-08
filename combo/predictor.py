import numpy as np
import pickle
from . import gp

class base_predictor( object ):
    def __init__( self, config, model = None ):
        self.config = config
        self.model = model
        if self.model is None:
            self.model = gp.core.model(cov = gp.cov.gauss( num_dim = None, ard = False ), mean = gp.mean.const(), lik = gp.lik.gauss())

    def fit( self, *args, **kwds ):
        raise NotImplementedError

    def prepare( self, *args, **kwds ):
        raise NotImplementedError

    def delete_stats( self, *args, **kwds ):
        raise NotImplementedError

    def get_basis( self, *args, **kwds ):
        raise NotImplementedError

    def get_post_fmean( self, *args, **kwds ):
        raise NotImplementedError

    def get_post_fcov( self, *args, **kwds ):
        raise NotImplementedError

    def get_post_params( self,*args, **kwds ):
        raise NotImplementedError

    def get_post_samples( self, *args, **kwds ):
        raise NotImplementedError

    def get_predict_samples( self, *args, **kwds ):
        raise NotImplementedError

    def get_post_params_samples( self, *args, **kwds ):
        raise NotImplementedError

    def update( self,*args, **kwds ):
        raise NotImplementedError

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.update(tmp_dict)
