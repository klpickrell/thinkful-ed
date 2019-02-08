#!/usr/bin/env python

import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt

def logp_ab(value):
    '''Gelman's uninformative prior'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

def _main():
    y = np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
         1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
         5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
        10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
        15,  9,  4
    ])
    n = np.array([
        20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
        20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
        46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
        48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
        47, 24, 14
    ])
    
    N = len(n)
#    trials = np.full((10,),100)
#    observed = np.array( [40,44,47,54,63,46,44,49,58,50] )
    with pm.Model() as model:
#        true_rates = pm.Beta('true_rates',a,b,size=10)
        ab = pm.HalfFlat('ab',shape=2,testval=np.asarray([1.,1.]))
        pm.Potential('p(a,b)',logp_ab(ab))

        X = pm.Deterministic('X',tt.log(ab[0]/ab[1]))
        Z = pm.Deterministic('Z',tt.log(tt.sum(ab)))

        theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)

#        n = trials
#        y = observed

        p = pm.Binomial('y', p=theta, observed=y, n=n)
        trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept': 0.95})
        pm.traceplot(trace, varnames=['ab','X','Z'])
        plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
