import numpy as np


class linear(object):
    def __init__(self, basis, params=None, bias=None):
        self.basis = basis
        self.nbasis = basis.nbasis
        self._init_params = params
        self.bias = bias
        self.params = params

        if params is None:
            self.params = np.zeros(self.nbasis)
        self.nparams = self.nbasis

    def get_mean(self, X, Psi=None, params=None, bias=None):
        if params is None:
            params = np.copy(self.params)

        if bias is None:
            bias = np.copy(self.bias)

        if Psi is None:
            Psi = self.get_basis(X)

        return Psi.dot(params) + bias

    def set_params(self, params):
        self.params = params

    def set_bias(self, bias):
        self.bias = bias

    def _init_params(self, params):
        if params is None:
            self.params = np.zeros(self.nbasis)

        self.params = params

    def _init_bias(self, bias):
        if bias is None:
            self.bias = 0

        self.bias = bias
