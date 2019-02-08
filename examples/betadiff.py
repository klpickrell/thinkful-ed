#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import beta, norm


def _main():
    n,k = 100,4
    plt.tight_layout()
    fig = plt.figure( figsize=(8,8) )
    fig.suptitle('Approximation error by sample size')
    for n in range(2,6):
        plt.subplot(2,2,n-1)
#        plt.title('n={}'.format(10**n))
        n = 10**n
        k = int(0.04*n)
        sns.distplot( beta.rvs( a=k+1, b=n-k+1, size=10000 ), color='blue', bins=200, label='beta', axlabel='n={}'.format(n) )
        sns.distplot( norm.rvs( k/n, np.sqrt( (k/n) * (1-(k/n)) * (1/n) ), size=10000 ), color='red', bins=200, label='normal', axlabel='n={}'.format(n) )
        plt.legend([ 'beta', 'normal' ])

    plt.savefig('betanormdiff.png'.format(n))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
