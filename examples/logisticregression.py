#!/usr/bin/env python

import math
import numpy as np
import random

class LogisticRegression(object):

    def __init__(self, alpha=0.01, miniters=10, maxiters=100, tolerance=0.5):
        self.alpha = alpha
        self.miniters = miniters
        self.maxiters = maxiters
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def fit( self, X, y, shuffle=True):
        ''' fit the model '''
        if (X is None or not len(X)) or (y is None or not len(y)):
            return self

        if shuffle:
            result = list(zip(X,y))
            random.shuffle(result)
            X,y = zip(*result)
            X,y = np.array(X),np.array(y)
    
        i, j, label, n_features, n_examples = (0,0,0,0,0)
        rate, rate_n, bias, predicted, update, error, logit = (0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    
        n_examples, n_features = X.shape
        rate = 0.01
    
        weights = X[0] * 0
        bias = 0
        n = self.maxiters
        for i in range(n):
            rate_n = rate * (n-i)/n
            tss = 0.0
            for _ in range(n_examples) :
                j = random.randrange(n_examples)
                label = y[j]
    
                logit = bias
                for k in range(n_features):
                    logit += weights[k] * X[j,k]
    
                predicted = 1.0 / (1.0 + math.exp(min(-logit,100)))
    
                error = label - predicted
                for k in range(n_features):
                    update = error * X[j,k] - (self.alpha * weights[k])
                    weights[k] += rate_n * update
    
                bias += rate_n * error
                tss += error * error
    
            if i > self.miniters and tss < self.tolerance:
                print( "convergence ({} < {})".format(tss, tolerance) )
                break
        print( "exit@{}  measured tss={} tolerance={}".format(i,tss,self.tolerance) )

        self.weights = weights
        self.bias = bias
        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise Exception( 'predict called without weights or bias (call fit first)' )

        logit = self.bias + np.dot(X, self.weights)
        predictions = 1.0 / (1.0 + np.exp(-logit))
        return np.round(predictions)




def _main():
    from sklearn.datasets import make_classification
    from sklearn.metrics  import classification_report

    X,y = make_classification( n_samples=1000, n_features=5, n_classes=2 )
    lr = LogisticRegression()
    lr.fit(X,y)
    predictions = lr.predict(X)
    print( classification_report( y, predictions ) )

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
