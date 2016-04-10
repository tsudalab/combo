import numpy as np
import core
from ..predictor import base_predictor


class predictor(base_predictor):
    def __init__(self, config, model=None):
        super(predictor, self).__init__(config, model)
        self.blm = None

    def fit(self, training, num_basis=None):
        if num_basis is None:
            num_basis = self.config.predict.num_basis

        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        self.model.fit(training.X, training.t, self.config)
        self.blm = self.model.export_blm(num_basis)
        self.delete_stats()

    def prepare(self, training):
        self.blm.prepare(training.X, training.t, training.Z)

    def delete_stats(self):
        self.blm.stats = None

    def get_basis(self, X):
        return self.blm.lik.get_basis(X)

    def get_post_fmean(self, training, test):
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_fmean(test.X, test.Z)

    def get_post_fcov(self, training, test):
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_fcov(test.X, test.Z)

    def get_post_params(self, training, test):
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_params_mean()

    def get_post_samples(self, training, test, N=1, alpha=1.0):
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.post_sampling(test.X, Psi=test.Z, N=N, alpha=alpha)

    def get_predict_samples(self, training, test, N=1):
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.predict_sampling(test.X, Psi=test.Z, N=N)

    def update(self, training, test):
        if self.model.stats is None:
            self.prepare(training)
            return None

        if hasattr(test.t, '__len__'):
            N = len(test.t)
        else:
            N = 1

        if N == 1:
            if test.Z is None:
                try:
                    test.X.shape[1]
                    self.blm.update_stats(test.X[0, :], test.t)
                except:
                    self.blm.update_stats(test.X, test.t)
            else:
                try:
                    test.Z.shape[1]
                    self.blm.update_stats(test.X[0, :], test.t, psi=test.Z[0, :])
                except:
                    self.blm.update_stats(test.X, test.t, psi=test.Z)
        else:
            for n in xrange(N):
                if test.Z is None:
                    self.blm.update_stats(test.X[n, :], test.t[n])
                else:
                    self.blm.update_stats(test.X[n, :], test.t[n], psi=test.Z[n, :])
