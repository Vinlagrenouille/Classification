import numpy as np
import matplotlib.pyplot as plt

    #Parameters
    mu0 = np.array([0.0, 0.0])
    mu1 = np.array([3.0, 2.0])
    sigma = np.array([[1.0, 0.5],[0.5, 1.0]])
    
    #C0 and C1 classes generation
    c0 = np.random.multivariate_normal(mu0, sigma, 10) 
    c1 = np.random.multivariate_normal(mu1, sigma, 10)
    
    #C0 and C1 classes generation BIS
    c01 = np.random.multivariate_normal(mu0, sigma, 1000)
    c11 = np.random.multivariate_normal(mu01, sigma, 1000)

def show():
    
    plt.scatter(c0[:,0], c0[:,1], c = 'red')
    plt.scatter(c1[:,0], c1[:,1], c = 'blue')
    plt.show()
    
    plt.scatter(c01[:,0], c01[:,1], c = 'red')
    plt.scatter(c11[:,0], c11[:,1], c = 'blue')
    plt.show()
    
def adl():
        
    
show()

