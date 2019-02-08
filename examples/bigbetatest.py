#!/usr/bin/env python

import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns

def logp_ab(value):
    '''Gelman's uninformative prior'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

def _main():
    
    data = pd.read_csv('data/jbi.csv')
    names = pd.read_csv('data/names.csv')
    data['name'] = names
    data.set_index('name',inplace=True)
    data['Dashboard Traffic'] = data['Dashboard Traffic'].str.replace(',','').astype(int)
    data['Total Quality Leads'] = data['Total Quality Leads'].str.replace(',','').astype(int)
    data['Avg Leads/User']
    data['crate'] = data['Total Quality Leads']/data['Dashboard Traffic']

    n = data['Dashboard Traffic'].values
    y = data['Total Quality Leads'].values
    N = len(y)

    with pm.Model() as model:
        ab = pm.HalfFlat('ab',shape=2,testval=np.asarray([1.,1.]))
        pm.Potential('p(a,b)',logp_ab(ab))
        theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)
        p = pm.Binomial('y', p=theta, observed=y, n=n)
        trace = pm.sample(10000, tune=1000, nuts_kwargs={'target_accept': 0.95}, cores=8)
        plt.figure()
        pm.traceplot(trace, varnames=['ab'])
        plt.savefig('theta_trace.png')
        plt.figure()
        pm.plot_posterior(trace)
        plt.savefig('posterior_trace.png')
        
        plt.figure()
        posteriors = []
        for i in range(N):
            posteriors.append( trace['theta'][:][:,i])
        for posterior in posteriors:
            sns.kdeplot(posterior,shade=True)

        plt.savefig( 'posterior_trace_sns.png' )
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
