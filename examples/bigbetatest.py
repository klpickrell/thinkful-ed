#!/usr/bin/env python

import pandas as pd
import numpy as np
import altair as alt
from collections import Counter,defaultdict
from itertools import combinations
from tqdm import tqdm
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns

import pickle
import os
from pprint import pprint as pp

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from tqdm import tqdm

def model_features(data):
    columns = [ 'JBI Homescreen VARIATION','JBI VH VARIATION','Your Plan Variation' ]
    xdata = data[columns].copy()
    featurenames1 = xdata['JBI Homescreen VARIATION'].unique() 
    featurenames2 = xdata['JBI VH VARIATION'].unique()
    featurenames3 = xdata['Your Plan Variation'].unique()
    xdata = pd.get_dummies(xdata)
    X = xdata.values
    y = data.performance
    
    ranks = []
    for i in tqdm(range(50)):
        Xn,yn = shuffle(X,y,random_state=np.random.randint(0,10000000))
        clf = RandomForestRegressor(n_estimators=100)
        clf.fit(Xn,yn)
        ranki = [ i[0] for i in reversed(sorted(zip(xdata.columns,clf.feature_importances_), key=lambda x: x[1])) ]
        ranks.append(ranki)

    cnts = defaultdict(int)
    for rk in ranks:
        for i,r in enumerate(rk):
            cnts[r] = cnts[r]+i

    pp( list(sorted(cnts.items(),key=lambda x:x[1])) )

def logp_ab(value):
    '''Gelman's uninformative prior'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

def _main():
    
    filename = 'jbi'
    data = pd.read_csv(os.path.join('data','{}.csv'.format(filename)))
    names = pd.read_csv('data/names.csv')
    data['name'] = names
    i_to_name = { i : data['name'].iloc[i] for i in range(len(names)) }
    data['Dashboard Traffic'] = data['Dashboard Traffic'].str.replace(',','').astype(int)
    data['Total Quality Leads'] = data['Total Quality Leads'].str.replace(',','').astype(int)
#    data['Avg Leads/User']
    data['crate'] = data['Total Quality Leads']/data['Dashboard Traffic']
    print( 'Total Quality Leads (mean): {}'.format(data['Total Quality Leads'].mean()) )
    print( 'Dashboard Traffic (mean): {}'.format(data['Dashboard Traffic'].mean()) )
    dfsmall = data[['name','JBI Homescreen VARIATION','JBI VH VARIATION','Your Plan Variation']].copy()
    dfsmall.to_csv('namemap.csv',index=False)

#    performance = [ 17,0,0,0,5,9,4,5,18,18,13,6,19,11,0,21,0,0,0,0,2,0,15,5,0,0,6,0,13,0 ]
    performance = [ 2,1,12,0,7,11,12,1,5,0,12,13,1,4,5,12,29,2,1,0,1,2,16,2,5,1,12,1,1,1 ]
    data['performance'] = performance

#    model_features(data)
#    return

#    data.set_index('name',inplace=True)

    n = data['Dashboard Traffic'].values
    y = data['Total Quality Leads'].values
    N = len(y)

    n_samples, n_tune = 50000,5000
    model_file = 'model_{}_{}.pkl'.format(n_samples,filename)
    if not os.path.exists(model_file):
        with pm.Model() as model:
#            ab = pm.HalfFlat('ab',shape=2,testval=np.asarray([1.,1.]))
#            ab = pm.Uniform('ab',shape=2,testval=np.asarray([0.,1.]))
#            pm.Potential('p(a,b)',logp_ab(ab))
#            theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)
            alpha = int(data['Total Quality Leads'].mean())
            beta = int(data['Dashboard Traffic'].mean())
            print('alpha: {}'.format(alpha+1))
            print('beta: {}'.format(beta-alpha+1))
            theta = pm.Beta('theta', alpha=alpha+1, beta=beta-alpha+1, shape=N)
            p = pm.Binomial('y', p=theta, observed=y, n=n)
            trace = pm.sample(n_samples, tune=n_tune, nuts_kwargs={'target_accept': 0.95}, cores=8)
            with open( 'model_{}.pkl'.format(n_samples), 'wb' ) as fil:
                pickle.dump( {'model' : model, 'trace' : trace }, fil )
    else:
        with open( model_file, 'rb' ) as fil:
            r = pickle.load( fil )
            model = r['model']
            trace = r['trace']

    burn_trace = trace[n_tune:]
    results = []
    for i in range(N):
        row = []
        for j in range(N):
#            iwins = ((burn_trace['theta'][:,i] > burn_trace['theta'][:,j]).sum())/len(burn_trace['theta'][:,i])
            s1 = burn_trace['theta'][:,i]
            s2 = burn_trace['theta'][:,j]
            iwins = (s1-s2 > 0).mean()
            v = { 'name' : i_to_name[i], 'name2' : i_to_name[j], 'winpct' : '{:.3%}'.format(iwins) }
            row.append(v)
        results.extend(row)

    out = pd.DataFrame(results)
    out.set_index('name',inplace=True)
    out.to_csv('comparison.csv')

    plot = True
    if plot:
        plt.figure()
#        pm.traceplot(burn_trace, varnames=['ab'])
        pm.traceplot(burn_trace)
        plt.savefig('theta_trace.png')
        plt.figure()
        pm.forestplot(burn_trace, xtitle='95% credibles for conversion rate by experiment {}'.format(filename), chain_spacing=0.06,vline=1)
        plt.savefig('forest_trace.png')
        plt.figure()
        pm.plot_posterior(burn_trace)
        plt.savefig('posterior_trace.png')

        res = pm.stats.summary(burn_trace)
        res.to_csv('summary_stats_{}.csv'.format(filename))
        
        plt.figure()
        posteriors = []
        for i in range(N):
            posteriors.append( burn_trace['theta'][:,i])
        for posterior in posteriors:
            sns.kdeplot(posterior,shade=True)
    
        plt.savefig( 'posterior_trace_sns.png' )

    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
