import numpy as np
from scipy import stats
from scipy.special import erf


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


singleGaussianList = ["singleGaussian", evalSingleGaussian, 2, [r'$\mu$', r'$\sigma$']]
twoGaussianList = ["twoGaussian", evalTwoGaussian, 5, [r'$\mu_1$', r'$\mu_2$', r'$\sigma_1$', r'$\sigma_2$', r'$\alpha$']]
uniformList = ["uniform", evalUniform, 2, [r'$m_l$', r'$m_u$']]