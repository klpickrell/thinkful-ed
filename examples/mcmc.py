#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def _main():
    # update observation data
    obs = np.array( [ 100, 85, 92 ] )
    prior = norm(np.mean(obs),np.std(obs))
    samples = np.zeros(10000)
    samples[0] = 105
    for i in range(1,len(samples)):
        proposal = samples[i-1] + np.random.randn()*5
        if prior.pdf(proposal) / prior.pdf(samples[i-1]) > np.random.uniform():
            samples[i] = proposal
        else:
            samples[i] = samples[i-1]

    sns.distplot(samples, bins=200)
    plt.show()

    return 0



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
