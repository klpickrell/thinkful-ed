#!/usr/bin/env python

import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

def _main():
    data = pd.read_csv('data/sfcatviews.csv')
    target = (data['SF/Cat Page Views']/(data.sum(axis=1))).diff()[1]
    total = data.sum().sum()
    print( 'target proportion difference is {}'.format(target) )
    phat = data['SF/Cat Page Views'].sum() / total
    print( 'estimated P: {}, 1-P: {}'.format(phat,1-phat) )
    
    n1 = data.sum(axis=1)[0]
    n2 = data.sum(axis=1)[1]
    p1 = (data['SF/Cat Page Views']/n1)[0]
    p2 = (data['SF/Cat Page Views']/n2)[1]
    zstar = (p1-p2)/np.sqrt( ((p1*(1-p1)/n1) + ((p2*(1-p2)/n2) )) )
    print( '''n1 = data.sum(axis=1)[0]\nn2 = data.sum(axis=1)[1]\np1 = (data['SF/Cat Page Views']/n1)[0]\np2 = (data['SF/Cat Page Views']/n2)[1]\nzstar = (p1-p2)/np.sqrt( ((p1*(1-p1)/n1) + ((p2*(1-p2)/n2) )) )''')
    print( 'z statistic: {}'.format(zstar) )

    import pdb; pdb.set_trace()
#    edata = pd.read_csv('data/experiment1.csv')
#    edata['mprops'].plot.hist(bins=20)
#    plt.savefig( 'samplingdistribution.png' )
#    chart = alt.Chart(edata).mark_bar().encode(x=alt.X('mprops:Q',
#                                                       axis=alt.Axis(title='Measured Difference in Proportions'),
#                                                       bin=alt.Bin(maxbins=100)),
#                                               y=alt.Y('count(mprops):Q',axis=alt.Axis(title='Number of Experiments')))
#    chart.serve()

    experiment = False
    if experiment:
        print( 'running 1000 experiments with {} examples'.format(total) )
        mprops = []
        for i in tqdm(range(1000)):
            rns = np.random.randint(1,5,total)
            cnt = Counter(rns)
            mprop = (cnt[1]/float(cnt[1]+cnt[2]))-(cnt[3]/float(cnt[3]+cnt[4]))
            mprops.append(mprop)

        pd.DataFrame( { 'mprops' : mprops } ).to_csv('data/experiment1.csv')
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
