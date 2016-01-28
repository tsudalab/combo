try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np

class res:
    def __init__( self, max_iter, directory = 'res', options = () ):
        self.nsearch = 0
        self.queries = np.zeros( max_iter )
        self.ttrain = np.empty(0)
        self.query_time = np.zeros( max_iter )
        self.best_ttrain = np.zeros( max_iter )
        self.best_t = - np.inf
        self.directory = directory
        self.options = options

        if not os.path.isdir(self.directory):
            os.mkdir( directory )

    def set_ttrain( self, ttrain ):
        self.ttrain = ttrain

    def write( self, query, t ):
        self.queries[ self.nsearch ] = query
        self.ttrain = np.append( self.ttrain, t )
        if t> self.best_t:
            self.best_t = t
        self.best_ttrain[ self.nsearch ] = self.best_t
        self.nsearch +=1

    def save( self, process = 0 ):
        filename = self.directory + '/res_bayes_search%03d.txt' %( process )
        f = open(filename,'w')
        pickle.dump( self, f )
        f.close()
