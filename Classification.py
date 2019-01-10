import numpy as np
import matplotlib.pyplot as plt

#Parameters
mu0 = np.array([0.0, 0.0])
mu1 = np.array([3.0, 2.0])
sigma = np.array([[1.0, 0.5],[0.5, 1.0]])

#C0 and C1 classes generation
np.random.seed(0)
c0 = np.random.multivariate_normal(mu0, sigma, 10) 
c1 = np.random.multivariate_normal(mu1, sigma, 10)

#Training
c01 = np.random.multivariate_normal(mu0, sigma, 1000)
c11 = np.random.multivariate_normal(mu1, sigma, 1000)

#Testing
t0 = np.random.multivariate_normal(mu0, sigma, 100)
t1 = np.random.multivariate_normal(mu1, sigma, 100)

def show():
    
    plt.scatter(c0[:,0], c0[:,1], c = 'red')
    plt.scatter(c1[:,0], c1[:,1], c = 'blue')
    plt.show()
    
    plt.scatter(c01[:,0], c01[:,1], c = 'red')
    plt.scatter(c11[:,0], c11[:,1], c = 'blue')
    plt.show()
    
def moyenne(c):
    mean = np.mean(c, axis = 0)
    return np.array([[mean[0]], [mean[1]]])
    
def covMatrix(c):
    sigma = np.cov(np.transpose(c))
    return sigma
    
def probaDeC(c, d):
    return c.size/(c.size + d.size)
    
def sIGMA(sigma1, size1, sigma2, size2):
    return np.divide(np.add(np.dot(sigma1,size1),np.dot(sigma2, size2)),np.add(size1,size2))
    
def adl(sigInv, cmu0, cmu1, pi0, pi1, x):
    decision = np.add(np.subtract(np.dot(np.dot(np.transpose(x), sigInv), (cmu0 - cmu1)), np.dot(np.dot(np.dot(0.5, np.transpose(cmu0 - cmu1)), sigInv), (cmu0 + cmu1))), np.log(np.divide(pi0, pi1)))
    return decision
    
    
cmu0 = moyenne(c01)
cmu1 = moyenne(c11)

pi0 = probaDeC(c01, c11)
pi1 = probaDeC(c11, c01)

csigma0 = covMatrix(c01)
csigma1 = covMatrix(c11)

sig = sIGMA(csigma0, c01.size, csigma1, c11.size)
sigInv = np.linalg.inv(sig)    
    
adl(sigInv, cmu0, cmu1, pi0, pi1, c0)


    

