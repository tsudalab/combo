# -*- coding:utf-8 -*-
import numpy as np


class fourier(object):
    ''' class for fourier basis '''
    def __init__(self, params):
        self._check_params(params)
        self._check_len_params(params)
        self.params = params
        self.nbasis = self.params[1].shape[0]

    def get_basis(self, X, params=None):
        ''' compute basis functions '''
        if params is None:
            params = self.params

        self._check_params(params)
        self._check_len_params(params)

        return np.cos(np.dot(X, params[0].transpose()) + params[1]) * params[2]

    def set_params(self, params):
        ''' substitute the 3-dimensional
            tuple <params> into the instance <params> '''
        self._check_params(params)
        self._check_len_params(params)
        self.params = params

    def show(self):
        print('W = ', self.params[0])
        print('b = ', self.params[1])
        print('alpha = ', self.params[2])

    def _check_params(self, params):
        if not isinstance(params, tuple):
            raise ValueError('The variable < params > must be a tuple.')

        if len(params) != 3:
            message = "The variable < params > must be 3-dimensional tuple."
            raise ValueError(message)

    def _check_len_params(self, params):
        if params[0].shape[0] != params[1].shape[0]:
            message =\
                "The length of 0-axis of W must be same as the length of b."
            raise ValueError(message)

        if hasattr(params[2], "__len__"):
            if len(params[2]) != 1:
                message = "The third entry of <params> must be a scalar."
                raise ValueError(message)
            else:
                if isinstance(params[2], str):
                    message = "The third entry of <params> must be a scalar."
                    raise ValueError(message)
