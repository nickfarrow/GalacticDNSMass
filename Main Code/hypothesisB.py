import pymultinest
import numpy as np
from scipy import stats
from scipy.special import erf
import os
import sys
from datetime import datetime

startTime = datetime.now()

###
# Directory of current script
# Required by Condor jobs to find relevant samples etc.
topDirectory = os.path.dirname(os.path.realpath(__file__)) + '/'

# Load BNS mass samples
# (eg. 17x10000 array)
pulsarSamples = np.loadtxt(topDirectory + 'Samples/mr_samples.txt')
companionSamples = np.loadtxt(topDirectory + 'Samples/ms_samples.txt')

# For each BNS, group pulsar & companion samples into pairs.
# Creates array of shape like (17x10000x2)
bothMassSamples = np.stack((pulsarSamples, companionSamples), axis=-1)

# Only use first 1000 samples to increase speed
massSamples = bothMassSamples[:,:1000,:]

# Define the nMeasurements and nSamples per mass measurement
nSamples = len(massSamples[0])
nMeasurements = len(massSamples)



### Probability Distribution Functions
# Single Gaussian Functions
def evalSingleGaussian(theta, x):
    mu, sig = theta[0], theta[1]
    normalisingTerm = (0.5*(1+erf((2-mu)/(sig*2**0.5)) - (1+erf((0.8-mu)/(sig*2**0.5)))))
    return stats.norm(mu, sig).pdf(x) * 1.0/normalisingTerm

# Two Gaussian Functions
def evalTwoGaussian(theta, x):
    mu1, mu2, sig1, sig2, alpha = theta
    normalisingTerm1 = (0.5*(1+erf((2-mu1)/(sig1*2**0.5)) - (1+erf((0.8-mu1)/(sig1*2**0.5)))))
    normalisingTerm2 = (0.5*(1+erf((2-mu2)/(sig2*2**0.5)) - (1+erf((0.8-mu2)/(sig2*2**0.5)))))
    return alpha * stats.norm(mu1, sig1).pdf(x) * 1.0/normalisingTerm1 + (1-alpha) * stats.norm(mu2, sig2).pdf(x) * 1.0/normalisingTerm2

# Uniform Functions
def evalUniform(theta, x):
    mMin, mMax = theta[0], theta[1]
    return stats.uniform(mMin, mMax-mMin).pdf(x)



### Model Information Dictionaries
singleGaussianModel = {
    "name": "singleGaussian",
    "pdf": evalSingleGaussian,
    "ndim": 2,
    "params": ['mu', 'sigma']
}

twoGaussianModel = {
    "name": "twoGaussian",
    "pdf": evalTwoGaussian,
    "ndim": 5,
    "params": ['mu1', 'mu2', 'sigma1', 'sigma2', 'alpha']
}

uniformModel = {
    "name": "uniform",
    "pdf": evalUniform,
    "ndim": 2,
    "params": ['mMin', 'mMax']
}



### Prior for each model.
def prior(cube, ndim, nparams):
    # cube is initially a unit hypercube which is to be mapped onto the relevant prior space.

    
    # Hyperparameter Index. We map the priors beginning with the parameters of model1,
    # and then incriment j by the number of parameters in that model1. Then mapping
    # the parameters of the next model2.
    j = 0
    
    # Loop over both models.
    for modelBeingMapped in [modelName1, modelName2]:
        if modelBeingMapped == 'singleGaussian':
            cube[j] = 0.8 + cube[j] * (2 - 0.8)
            cube[j+1] = 0.005 + cube[j+1] * (0.5 - 0.005)
            j += 2
        if modelBeingMapped == 'twoGaussian':
            cube[j] = 0.8 + cube[j] * (2 - 0.8)
            cube[j+1] = cube[j] + cube[j+1] * (2 - cube[j])
            cube[j+2] = 0.005 + cube[j+2] * (0.5 - 0.005)
            cube[j+3] = 0.005 + cube[j+3] * (0.5 - 0.005)
            cube[j+4] = cube[j+4] * 1
            j += 5
        if modelBeingMapped == 'uniform':
            cube[j] = 0.8 + cube[j] * (2 - 0.8)
            cube[j+1] = cube[j] + cube[j+1] * (2 - cube[j])
            j += 2
    
    # After this process the number of parameters mapped, j, should be equal to the number of dimensions for the problem (hyperparameters of model1 + model2).
    if j != ndim:
        print("SOME PARAMETERS WERE NOT MAPPED TO!!!")
        print(ndim)
        print(j)
    
    return



### Likelihood Function (Same as Farr et al.)
def likelihood(cube, ndim, nparams):
    
    # Create lists of the parameters for each model. Model1 has parameters in cube from 0 to ndim1-1, Model2 has parameters in cube from ndim1 to ndim-1.
    paramList1 = [cube[i] for i in range(ndim1)]
    paramList2 = [cube[i] for i in range(ndim1, ndim)]
    
    # Initial list to contain the sum of the products of the probability for each m_r and m_s sample in their respective models.
    pdfProductSumList = []
    
    # For the m_r and m_s pairs in each BNS system. (eg. 1000x2)
    for massSample in massSamples:
        
        # Evaluate the PDF function down the m_r and m_s samples of the BNS
        mrProbabilities = modelEval1(paramList1, massSample[:,0])
        msProbabilities = modelEval2(paramList2, massSample[:,1])
        
        # Evaluate the product of the m_r and m_s probability for each pair.
        probabilityProduct = mrProbabilities*msProbabilities
        
        # Append the sum over all the probability products of each pair.
        pdfProductSumList.append(np.sum(probabilityProduct))
    
    # If either all the m_r or all the m_s samples are completely outside their model then return a log-likelihood of -inf.
    if 0 in pdfProductSumList:
        print("Zero probability value - Parameters: {}, {}".format(paramList1,paramList2))
        return -np.inf
 
    # The log-likelihood is the log of the normalised sum over the log of each pdfProductSum
    loglikelihood = nMeasurements * np.log(1.0/nSamples) + np.sum(np.log(pdfProductSumList))
    return loglikelihood



### Models
# This list contains the set of all model dictionary combinations
# Model names, pdf functions, nDimensions, ParameterNames
modelSet = [[singleGaussianModel, singleGaussianModel],
 [singleGaussianModel, twoGaussianModel],
 [singleGaussianModel, uniformModel],
 [twoGaussianModel, singleGaussianModel],
 [twoGaussianModel, twoGaussianModel],
 [twoGaussianModel, uniformModel],
 [uniformModel, singleGaussianModel],
 [uniformModel, twoGaussianModel],
 [uniformModel, uniformModel]]


### System input (models choice)
# Takes the system argument given.
# eg hypothesisB.py 2 will select the model pair with index 1, singleGuassian and twoGaussian.
modelSetIndex = int(sys.argv[1]) - 1

# Set relevant variables which are to be used in this sampling
modelDictionary1, modelDictionary2 = modelSet[modelSetIndex]
modelName1, modelEval1, ndim1, paramNames1 = modelDictionary1['name'], modelDictionary1['pdf'], modelDictionary1['ndim'], modelDictionary1['params']
modelName2, modelEval2, ndim2, paramNames2 = modelDictionary2['name'], modelDictionary2['pdf'], modelDictionary2['ndim'], modelDictionary2['params']

# Combined parameter list
paramNames = paramNames1 + paramNames2

# Define the total number of parameters
ndim = ndim1 + ndim2



### Inference
# Directory to send output to. Create it if it does not exist.
directoryName = topDirectory + 'hypB_out/' + modelName1[:4] + "/" + modelName2[:4]
if not os.path.exists(directoryName):
    os.makedirs(directoryName)

# Run pymultinest sampling
pymultinest.run(likelihood, prior, ndim, n_live_points=100, sampling_efficiency=0.3, importance_nested_sampling=False, outputfiles_basename=directoryName + '/', resume=False, verbose=True)

print("Code took {} seconds to run.".format(datetime.now() - startTime))
print("Exiting!")

