import numpy as np
class Welford(object):
    def __init__(self):
        self.k = 0
        self.M = 0
        self.S = 0
        self.shape = (0,0)
    
    def update(self,x):
        if x is None:
            return
        nr, nc = x.shape
        if (self.shape == (0,0)):
            self.shape = (1,nc)
        
        if (not (self.shape[1] == nc)):
            return
            
        for i in range(0,nr):
            self.k += 1
            newM = self.M + (x[i,:] - self.M)*1./self.k
            newS = self.S + (x[i,:] - self.M)*(x[i,:] - newM)
            self.M, self.S = newM, newS
            
    @property
    def mean(self):
        return self.M
    @property
    def std(self):
        if (self.k==1):
            return 0
        return np.sqrt(self.S/(self.k-1))
    @property
    def count(self):
        return self.k