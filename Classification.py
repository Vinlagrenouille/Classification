import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Parameters
mu0 = np.array([0.0, 0.0])
mu1 = np.array([3.0, 2.0])
sigma = np.array([[1.0, 0.5],[0.5, 1.0]])

#C0 and C1 classes generation
np.random.seed(1)
c0 = np.random.multivariate_normal(mu0, sigma, 10) 
c1 = np.random.multivariate_normal(mu1, sigma, 10)

#Training
c01 = np.random.multivariate_normal(mu0, sigma, 1000)
c11 = np.random.multivariate_normal(mu1, sigma, 1000)

#Testing
t0 = np.random.multivariate_normal(mu0, sigma, 100)
t1 = np.random.multivariate_normal(mu1, sigma, 100)

def showLearn():
    plt.scatter(c0[:,0], c0[:,1], c = 'red')
    plt.scatter(c1[:,0], c1[:,1], c = 'blue')
    
def showTest():
    plt.scatter(c01[:,0], c01[:,1], c = 'red')
    plt.scatter(c11[:,0], c11[:,1], c = 'blue')
    
def moyenne(c):
    mean = np.mean(c, axis = 0)
    return np.array([[mean[0]], [mean[1]]])
    
def covMatrix(c):
    sigma = np.cov(np.transpose(c))
    return sigma
    
def probaDeC(c, d):
    return c.size/(c.size + d.size)
    
def sIGMA(sigma1, size1, sigma2, size2,lamb=2):
    if(lamb == 2):
        return np.divide(np.add(np.dot(sigma1,size1),np.dot(sigma2, size2)),np.add(size1,size2))
    return lamb * np.add(np.dot(size2/(size1+size2), sigma1),np.dot(size1/(size1+size2),sigma2)) + (1 - lamb) * np.identity(2) 
    
def adl(sigInv, cmu0, cmu1, pi0, pi1, x):
    a1 = np.dot(np.dot(x, sigInv), (cmu0 - cmu1))
    a2 = np.dot(np.dot(np.dot(0.5, np.transpose(cmu0 - cmu1)), sigInv), (cmu0 + cmu1))
    decision = np.add(np.subtract(a1, a2), np.log(np.divide(pi0, pi1)))
    w = np.dot(sigInv, np.subtract(cmu0, cmu1))
    return decision, w
    
def nblocksValidation(c0, c1):
    lambs = np.arange(0, 1.1,0.1)
    pred = [0]*11
    for i in range(len(lambs)):
        l = lambs[i]
        varSig = sIGMA(csigma0, c0.size, csigma1, c1.size,l)
        varSigInv = np.linalg.inv(varSig)
        for j in range(len(c0)):
            tmp0 = c0
            np.delete(tmp0,j)
            learn0, w = adl(varSigInv, cmu0, cmu1, pi0, pi1, j)
            if learn0[0][0] > 0:
                pred[i] += 1
        for k in range(len(c1)):
            tmp1 = c1
            np.delete(tmp1,k)
            learn1, w = adl(varSigInv, cmu0, cmu1, pi0, pi1, k)
            if learn1[0][0] < 0:
                pred[i] += 1
        pred[i] = pred[i]/(len(c0)+len(c1))
    max = 0
    for i in range(len(pred)):
        if(pred[i] > max):
            max = pred[i]
    return lambs[i]
        
    
#Parameters calculation    
cmu0 = moyenne(c0)
cmu1 = moyenne(c1)

pi0 = probaDeC(c0, c1)
pi1 = probaDeC(c1, c0)

csigma0 = covMatrix(c0)
csigma1 = covMatrix(c1)

sig = sIGMA(csigma0, c0.size, csigma1, c1.size)
sigInv = np.linalg.inv(sig)    

#ADL results
learn0, w = adl(sigInv, cmu0, cmu1, pi0, pi1, c0)
learn1, w = adl(sigInv, cmu0, cmu1, pi0, pi1, c1)
learn01, w = adl(sigInv, cmu0, cmu1, pi0, pi1, c01)
learn11, w = adl(sigInv, cmu0, cmu1, pi0, pi1, c11)

#Classification rates
learnTP = 0
learnTN = 0
learnFP = 0
learnFN = 0
for r in np.nditer(learn0):
    if 0 < r:
        learnTP += 1
    else:
        learnFN += 1
for r in np.nditer(learn1):
    if 0 < r:
        learnFP += 1
    else:
        learnTN += 1
learnAcc = (learnTP+learnTN)/20
print("Apprentissage Accuracy = ",learnAcc)

testTP = 0
testTN = 0
testFP = 0
testFN = 0
for r in np.nditer(learn01):
    if 0 < r:
        testTP += 1
    else:
        testFN += 1
for r in np.nditer(learn11):
    if 0 < r:
        testFP += 1
    else:
        testTN += 1
testAcc = (testTP+testTN)/2000
print("Test Accuracy = ",testAcc)

#SkLearn job
ensL = np.concatenate((c0, c1), axis=0)
class0 = np.zeros((10,), dtype=int)
class1 = np.ones((10,), dtype=int)
classL = np.concatenate((class0, class1), axis=0)
skLDAlearn = LDA()
skLDAlearn.fit(ensL,classL)
pred0 = skLDAlearn.predict(c01) 
pred1 = skLDAlearn.predict(c11) 
skTP = 0
skTN = 0
skFP = 0
skFN = 0
for r in np.nditer(pred0):
    if 0 == r:
        skTP += 1
    else:
        skFN += 1
for r in np.nditer(pred1):
    if 0 == r:
        skFP += 1
    else:
        skTN += 1
skAcc = (skTP+skTN)/2000
print("SKLearn Accuracy = ",skAcc)
    
#Change first observation of C0
ctmp0 = np.copy(c0)
ctmp0[0] = [-10,-10]
c0mu0 = moyenne(ctmp0)
c0mu1 = moyenne(c1)
c0sigma0 = covMatrix(ctmp0)
c0sigma1 = covMatrix(c1)
sig0 = sIGMA(c0sigma0, ctmp0.size, c0sigma1, c1.size)
sigInv0 = np.linalg.inv(sig)
learn0, w = adl(sigInv0, c0mu0, c0mu1, pi0, pi1, ctmp0)
learn1, w = adl(sigInv0, c0mu0, c0mu1, pi0, pi1, c1)
learn01, w = adl(sigInv0, c0mu0, c0mu1, pi0, pi1, c01)
learn11,w = adl(sigInv0, c0mu0, c0mu1, pi0, pi1, c11)
learnTP = 0
learnTN = 0
learnFP = 0
learnFN = 0
for r in np.nditer(learn0):
    if 0 < r:
        learnTP += 1
    else:
        learnFN += 1
for r in np.nditer(learn1):
    if 0 < r:
        learnFP += 1
    else:
        learnTN += 1
learnAcc = (learnTP+learnTN)/20
print("Apprentissage Accuracy = ",learnAcc)

testTP = 0
testTN = 0
testFP = 0
testFN = 0
for r in np.nditer(learn01):
    if 0 < r:
        testTP += 1
    else:
        testFN += 1
for r in np.nditer(learn11):
    if 0 < r:
        testFP += 1
    else:
        testTN += 1
testAcc = (testTP+testTN)/2000
print("Test Accuracy = ",testAcc)

b = np.dot(-0.5, np.dot(np.transpose(np.subtract(cmu0,cmu1)),np.dot(sigInv,np.add(cmu0, cmu1)))) + np.log(pi0 / pi1)
point_x = np.transpose([0, -b / w[1]])
point_y = np.transpose([-b / w[0], 0])
plt.plot(point_x, point_y)
plt.scatter(ctmp0[:,0], ctmp0[:,1], c = 'red')
plt.scatter(c1[:,0], c1[:,1], c = 'blue')
plt.show()

#Decision boundary
b = np.dot(-0.5, np.dot(np.transpose(np.subtract(cmu0,cmu1)),np.dot(sigInv,np.add(cmu0, cmu1)))) + np.log(pi0 / pi1)
point_x = np.transpose([0, -b / w[1]])
point_y = np.transpose([-b / w[0], 0])
plt.plot(point_x, point_y)
showLearn()
plt.show()

#LDA variant
resultLearn = []
resultTest = []
lambs = []
for i in range(11):
    l = i/10
    lambs.append(l)
    varSig = sIGMA(csigma0, c0.size, csigma1, c1.size,l)
    varSigInv = np.linalg.inv(varSig)
    learn0, w = adl(varSigInv, cmu0, cmu1, pi0, pi1, c0)
    learn1, w = adl(varSigInv, cmu0, cmu1, pi0, pi1, c1)
    learn01, w = adl(varSigInv, cmu0, cmu1, pi0, pi1, c01)
    learn11,w = adl(varSigInv, cmu0, cmu1, pi0, pi1, c11)
    learnTP = 0
    learnTN = 0
    learnFP = 0
    learnFN = 0
    for r in np.nditer(learn0):
        if 0 < r:
            learnTP += 1
        else:
            learnFN += 1
    for r in np.nditer(learn1):
        if 0 < r:
            learnFP += 1
        else:
            learnTN += 1
    learnAcc = (learnTP+learnTN)/20
    resultLearn.append(learnAcc)
    
    testTP = 0
    testTN = 0
    testFP = 0
    testFN = 0
    for r in np.nditer(learn01):
        if 0 < r:
            testTP += 1
        else:
            testFN += 1
    for r in np.nditer(learn11):
        if 0 < r:
            testFP += 1
        else:
            testTN += 1
    testAcc = (testTP+testTN)/2000
    resultTest.append(testAcc)

plt.plot(lambs,resultLearn)
plt.plot(lambs,resultTest, c = "red")
plt.show()

#N-blocks validation
print("Best lambda is : ", nblocksValidation(c0,c1))