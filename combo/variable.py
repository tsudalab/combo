import numpy as np


class variable(object):
    def __init__(self, X=None, t=None, Z=None):
        self.X = X
        self.Z = Z
        self.t = t

    def get_subset(self, index):
        temp_X = self.X[index, :] if self.X is not None else None
        temp_t = self.t[index] if self.t is not None else None
        temp_Z = self.Z[index, :] if self.Z is not None else None

        return variable(X=temp_X, t=temp_t, Z=temp_Z)

    def delete(self, num_row):
        self.delete_X(num_row)
        self.delete_t(num_row)
        self.delete_Z(num_row)

    def add(self, X=None, t=None, Z=None):
        self.add_X(X)
        self.add_t(t)
        self.add_Z(Z)

    def delete_X(self, num_row):
        if self.X is not None:
            self.X = np.delete(self.X, num_row, 0)

    def delete_t(self, num_row):
        if self.t is not None:
            self.t = np.delete(self.t, num_row)

    def delete_Z(self, num_row):
        if self.Z is not None:
            self.Z = np.delete(self.Z, num_row, 0)

    def add_X(self, X=None):
        if X is not None:
            if self.X is not None:
                self.X = np.vstack((self.X, X))
            else:
                self.X = X

    def add_t(self, t=None):
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        if t is not None:
            if self.t is not None:
                self.t = np.hstack((self.t, t))
            else:
                self.t = t

    def add_Z(self, Z=None):
        if Z is not None:
            if self.Z is None:
                self.Z = Z
            else:
                self.Z = np.vstack((self.Z, Z))

    def save(self, file_name):
        np.savez_compressed(file_name, X=self.X, t=self.t, Z=self.Z)

    def load(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        self.X = data['X']
        self.t = data['t']
        self.Z = data['Z']
