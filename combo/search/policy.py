import numpy as np

class policy( object ):
    def __init__( self, call_simulator, option ):
        self.is_hyparam_learn = option.getboolean( 'general', 'is_hypara_learn' )
        self.is_rand_search = option.getboolean( 'general', 'is_rand_search' )
        self.is_rand_search

    def get_reward( self, *arg, **kwarg ):
        return reward

    def get_
