#!/usr/bin/env python

import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt

def logp_ab(value):
    '''Gelman's uninformative prior'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

def _main():

    n = np.full((10,),100)
    y = np.array( [40,44,47,54,63,46,44,49,58,50] )
    N = len(y)

    with pm.Model() as model:
#        true_rates = pm.Beta('true_rates',a,b,size=10)
        ab = pm.HalfFlat('ab',shape=2,testval=np.asarray([1.,1.]))
        pm.Potential('p(a,b)',logp_ab(ab))

#        X = pm.Deterministic('X',tt.log(ab[0]/ab[1]))
#        Z = pm.Deterministic('Z',tt.log(tt.sum(ab)))

        theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)

#        n = trials
#        y = observed

        p = pm.Binomial('y', p=theta, observed=y, n=n)
        trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept': 0.95})
        pm.traceplot(trace, varnames=['ab'])#,'X','Z'])
        pm.plot_posterior(trace)
        plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
