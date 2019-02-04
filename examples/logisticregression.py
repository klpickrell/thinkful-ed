#!/usr/bin/env python

import math
import numpy as np
import random

class LogisticRegression(object):

    def __init__(self, alpha=0.01, miniters=10, maxiters=100, tolerance=0.5, C=None):
        self.alpha = alpha
        self.miniters = miniters
        self.maxiters = maxiters
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.C = C

    def fit( self, X, y, shuffle=True, randinit=False):
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
    
        if randinit:
            weights = np.random.randn(len(X[0]))
        else:
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
                    if self.C is not None:
                        update = error * X[j,k] - (self.alpha * (weights[k]+(self.C/n_examples * weights[k])))
                    else:
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
    from sklearn.metrics  import classification_report, accuracy_score
    from sklearn.linear_model import LogisticRegression as sklearn_lr
    from sklearn.preprocessing import StandardScaler

    X,y = make_classification( n_samples=1000, n_features=5, n_classes=2 )
    for i in range(5):
        lr = LogisticRegression(C=0.1,maxiters=300)
        lr.fit(X,y,randinit=True,shuffle=False)
        predictions = lr.predict(X)
        print( classification_report( y, predictions ) )
        print( accuracy_score( y, predictions ) )
        sv = ''
        for i,w in enumerate(lr.weights):
            sv += '{}x{} + '.format(w,i)
        sv += '{}'.format(lr.bias)
        print(sv)

    print('sklearn')
    lr = sklearn_lr()
    lr.fit(X,y)
    predictions = lr.predict(X)
    print( classification_report( y, predictions ) )
    print( accuracy_score( y, predictions ) )

    print( '-'*20 +'standardized data' + '-'*20 )
    X = StandardScaler().fit_transform(X)
    lr = LogisticRegression(C=0.1,maxiters=300)
    lr.fit(X,y)
    predictions = lr.predict(X)
    print( classification_report( y, predictions ) )
    print( accuracy_score( y, predictions ) )

    print('sklearn')
    lr = sklearn_lr()
    lr.fit(X,y)
    predictions = lr.predict(X)
    print( classification_report( y, predictions ) )
    print( accuracy_score( y, predictions ) )
    

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
