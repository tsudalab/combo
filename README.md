COMmon Bayesian Optimization Library ( COMBO )
========


# Required Packages ############################
* Python 2.7.x
* numpy  >=1.10
* scipy  >= 0.16
* Cython >= 0.22.1
* mpi4py >= 2.0 (optional)

# Install ######################################
	1. Download or clone the github repository, e.g.
		> git clone http://git.tsudalab.org/tsuyos-u/combo.git

	2. Run setup.py install
		> cd combo
		> python setup.py install

# Uninstall

	1. Delete all installed files, e.g.
		> python setup.py install --record file.txt
		> cat file.txt  | xargs rm -rvf


# Usage
After installation, you can launch the test suite from 'examples/nano/tutorial.ipynb'.

``` python
import numpy as np
import combo
try:
	import cPickle as pickle
except
	import pickle

''' load the set of finger prints of searching candidates '''
X = np.load( '6atom/X.npy' )
t = np.load( '6atom/t.npy' ); global t

''' normalization of X: mean( X,0 ) and std( X, 0 ) '''
X = combo.misc.centering( X )
print np.mean( X, 0 ), np.std( X, 0 )

''' function for calling the simulator ( need to customize ) '''
def call_sim( i ):
	return t[i]

''' define the covariance ( kernel ) in the GP prior '''
width = 3; scale = 1
alpha = np.log( width ); beta = np.log( scale )
params = np.array([ alpha, beta ])

''' Gaussian ARD kernel  '''
cov = combo.gp.cov.gauss( ndims = X.shape[1], params = params, ard = True )

''' define the mean in the GP prior '''
mu0 = 0
mean = combo.gp.mean.const( mu0 )

''' define the likelihood in the GP '''
std_lik = 1
gamma = np.log( std_lik )
lik = combo.gp.lik.gauss( params = gamma )

''' define the Gaussian process '''
gp = combo.gp.model( mean = mean, cov = cov, lik = lik )

''' function for reseting the initial parameters of GP when '''
def reset_init_params( gp, X, t ):
    lik_params = np.log( 1 )
		width = 3; scale = np.std(t)
    cov_params = np.ones( gp.prior.cov.ndims ) * np.log( width )
    cov_params = np.append( cov_params, np.log( scale ) )
    mean_params = np.median( t )

    params = np.append( lik_params, mean_params )
    params = np.append( params, cov_params )

    ''' reset the initial hyper parameters '''
    gp.set_params( params )

''' set option for searching '''
search_options = {}

''' number of initial random sampling '''
search_options['nburn_in'] = 50

''' flag for hyper parameter learning. If true,
		hyper parameter is learned by means of the type 2 ML '''
search_options['learn'] = True  

''' learn the hyper parameters at each 'learn_timing'-step '''
search_options['learn_timing'] = 25

'''  flag for using kernel expansion '''
search_options['ker_expans'] = True

''' the number of basis functions '''
search_options['nbasis'] = 5000    

''' determine the scoring function default is TS '''
search_options['score'] = 'TS'  

''' directory name for saving the search results '''
search_options['directory_name'] = 'res'

''' maximum number of searching '''
search_options['max_iter'] = 300

''' flag for showing the process for searching '''
search_options['disp'] = True

''' set the initial parameters when learning the hyper parameters. \
If the key 'reset_init_params' is None, the initial parameter is set to the current hyper parameter.  '''
search_options['reset_init_params'] = reset_init_params

''' set option for learning '''
learn_options ={'method':'adam'}

''' display the learning process  '''
learn_options['disp'] = True

''' maximum number of learning epochs '''
learn_options['max_epoch'] = 3000   

''' interval of display '''
learn_options['interval_disp'] =1000

''' number of subset of samples for computing  '''
learn_options['subN_learn'] = 64   

''' number of subset of samples for evaluating the marginal likelihood '''
learn_options['subN_eval'] = 1024  



''' define the class for searching:
 call_sim: function for calling the simulation
 search_options: option for searching
 learnin_options: option for learning '''
bayes_search = combo.search.bayes( call_sim, search_options, learn_options )

''' set the seed parameter  '''
bayes_search.set_seed( 0 )

''' start searching
bayes_search.run( gp, Xtest, Xtrain = None , ttrain = None,  process = None )
gp: combo.gp.model : class for Gaussian process
Xtest: set of search candidates
Xtrain: initial data set ( default is None )
ttrain: initial data set ( default is None)
process: define the process number. 'process' is just used
for file  name to save the search results

searching results can be seen in the class bayes_search.res.
'''
bayes_search.run( gp, X )

''' bayes_search.res is automatically saved  in 'directory_name'/bayes_search_res_'process'.txt by using Pickle '''

''' if you do not define 'process',  'directory_name'/bayes_search_res_'seed'.txt '''

''' if you donot define the both of 'process' and 'seed',
the file name is  'directory_name'/bayes_search_res_000.txt '''

''' if you want the load search results '''
f = open('res/res_bayes_search000.txt')
res = cPickle.load(f)
print res.ttrain
print res.best_ttrain
f.close()

```   
