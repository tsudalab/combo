import numpy as np


class variable(object):
    def __init__(self, X=None, t=None, Z=None):
        self.X = X
        self.Z = Z
        self.t = t

    def delete(self, num_row):
        self.delete_x(num_row)
        self.delete_t(num_row)

        if self.Z is not None:
            self.delete_z(num_row)

    def add(self, new_x, new_t, new_z=None):
        self.add_x(new_x)
        self.add_t(new_t)

        if new_z is not None:
            self.add_z(new_z)


    def get_mini_batch(self, mini_batch_size):
        index = np.random.permutation( xrange( self.X.shape[0] ) )
        return self.X[index[0:mini_batch_size],:], self.t[index[0:mini_batch_size]]

    def split( self, N ):
        train_X, test_X = np.split( self.X, [ N ])
        train_t, test_t = np.split( self.t, [ N ])
        training = variable( X = train_X, t = train_t  )
        test = variable( X = test_X, t = test_t )
        return training, test

    def random_split( self, N ):
        index = np.random.permutation( xrange(self.X.shape[0]) )
        training = variable(X = self.X[index[0:N],:], t=self.t[index[0:N]])
        test = variable(X = self.X[index[N:],:], t=self.t[index[N:]])
        return training, test

    def delete_x(self, num_row):
        self.X = np.delete(self.X, num_row, 0)

    def delete_t(self, num_row):
        self.t = np.delete(self.t, num_row)

    def delete_z(self, num_row):
        self.Z = np.delete(self.Z, num_row, 0)

    def add_x(self, new_x):
        if self.X is None:
            self.X = new_x
        else:
            self.X = np.vstack((self.X, new_x))

    def add_t(self, new_t):
        if not isinstance(new_t, np.ndarray):
            new_t = np.array([new_t])

        if self.t is None:
            self.t = new_t
        else:
            self.t = np.hstack((self.t, new_t))

    def add_z(self, new_z):
        if self.Z is None:
            self.Z = new_z
        else:
            self.Z = np.vstack((self.Z, new_z))

    def save( self, file_name ):
        np.savez_compressed( file_name, X = self.X, t = self.t, Z = self.Z )

    def load( self, file_name ):
        data = np.load( file_name )
        self.X = data['X']
        self.t = data['t']
        self.Z = data['Z']
