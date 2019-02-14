#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm,expon,binom
from tqdm import tqdm


def target(lik,prior,n,h,theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return lik(n, theta).pmf(h) * prior.pdf(theta)

def _main():
    # update observation data
    obs1 = np.random.randn(1000)*20 + 100
    obs2 = np.random.randn(1000)*30 + 200
    obs = np.hstack((obs1,obs2))
    prior = norm(np.mean(obs),np.std(obs))
    n = 10000
    np.random.shuffle(obs)
    samples = np.hstack((obs,np.zeros(n)))
    sigma,theta = 0.3,0.1
    for i in tqdm(range(1,len(samples))):
        theta_p = theta + norm(0,sigma).rvs()
        rho = min(1, target(expon, prior, n, h, theta_p)/target(binom, prior,n,h,theta))
        if np.random.uniform() < rho:
            theta = theta_p
        samples[i+1] = theta

#        proposal = samples[i-1] + np.random.randn()*50
#        proposal = samples[i-1] + expon(samples[i-1]).rvs(1)[0] * np.random.choice([1,-1])
#        a,b = prior.pdf(proposal), prior.pdf(samples[i-1])
#        C_ratio = min(1,a/b)
#        if C_ratio > np.random.uniform():
#            samples[i] = proposal
#        else:
#            samples[i] = samples[i-1]
    import pdb; pdb.set_trace()
    burn = 1000
    sns.distplot(samples[burn:], bins=200)
    plt.axvline(np.mean(obs1))
    plt.axvline(np.mean(obs2))
    plt.show()

    return 0



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
