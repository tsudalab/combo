import numpy as np

class zero:
    ''' zero '''
    def __init__(self):
        self.num_params = 0
        self.params = np.array([])

    def get_mean(self, num_data, params = None):
        return np.zeros(num_data)

    def get_grad( self, num_data, params = None):
        return np.array([])

    def set_params( self, params ):
        pass
