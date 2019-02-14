#!/usr/bin/env python

import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

def logp_ab(value):
    '''Gelman's uninformative prior'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

def _main():
    y = np.hstack( (norm.rvs(100)*20+100, norm.rvs(100)*30+200) )
    N = len(y)
    with pm.Model() as model:
        ab = pm.HalfFlat('ab',shape=2,testval=np.asarray([1.,1.]))
        pm.Potential('p(a,b)',logp_ab(ab))

        mu = pm.Normal( 'mu', mu=0, sd=1 )
        obs = pm.Normal( 'obs', mu=0, sd=1, observed=y )
        trace = pm.sample(1000, tune=500, nuts_kwargs={'target_accept': 0.95})

        pm.traceplot(trace, varnames=['mu', 'ab'])
        
        plt.figure()
        sns.kdeplot(trace['mu'],shade=True)

        plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
