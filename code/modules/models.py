import numpy as np
import pymc3 as pm

def get_varnames(trace):
    """ return a list of variable names in a trace """
    dl = []
    for _, val in enumerate(trace.varnames):
        dl += [val.split("_")[0]]
    # remove duplicate names after split
    varn = set(dl)
    # remove deterministic variable y
    varn.remove('y')
    return list(varn)

def model_gauss(observations, xvalues, npeaks, *args, **kwargs):
    """ basic probabilistic model: the data is assumed to consist of three components
                                   1. an addition of M Gaussian shaped peaks
                                   2. a baseline which can be of shape: 
                                      none, y-offset or linear
                                   3. a residu consisting of zero centered, 
                                      normally distributed noise
    """
    """ parameters:
            observations: numpy array containing the y-values of the 
                          observations, with each row containg a single observation
            xvalues     : numpy array containing the x-values of the data (features)
            npeaks      : number of peaks presumed present in the data
            
        optional parameters:
            mu_peaks    : list of testvalues to use for the peak location 
                          (default: linearly spaced)
            pmodel      : the distribution uses for the peak locations 
                          (default: uniform)
            baseline    : baseline assumed present in the data (default: none)
                
            returns     : initialized pymc3 model object
    """
    
    mu_peaks = kwargs.get('mu_peaks', None)
    pmodel = kwargs.get('pmodel', None)
    baseline = kwargs.get('baseline', None)
    
    # maximum peak amplitude
    max_amp = observations.max()
    
    # the min and max peak values in the lognormal model can be shifted by 10% of the minimum value
    min_xval = xvalues.min() * 0.9
    max_xval = xvalues.max() + (xvalues.min() * 0.1)

    with pm.Model() as model:
        # priors for Gaussian peak shapes
        amp = pm.Uniform('amp', 0, max_amp, shape=(1, npeaks))

        if mu_peaks is not None:
            #print("mu_peaks: ", mu_peaks)
            if pmodel == 'normal':
                # use Normal model
                mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
                               shape=(1, npeaks), transform=pm.distributions.transforms.ordered, testval=mu_peaks)
            else:
                # use LogNormal model
                #mu = pm.Uniform('mu', xvalues.min(), xvalues.max(), shape=(1, npeaks), testval=mu_peaks)
                mu = pm.Uniform('mu', min_xval, max_xval, shape=(1, npeaks), testval=mu_peaks)
        else:
            if pmodel == 'normal':
                # use Normal model
                mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
                               shape=(1, npeaks), transform=pm.distributions.transforms.ordered)
            else:
                # use LogNormal model
                #mu = pm.Uniform('mu', xvalues.min(), xvalues.max(), shape=(1, npeaks))
                mu = pm.Uniform('mu', min_xval, max_xval, shape=(1, npeaks))

        if pmodel == 'normal':
            # use Normal model
            sigma = pm.HalfNormal('sigma', sd=100, shape=(1, npeaks))
        else:
            # use LogNormal model
            sigma = pm.Lognormal('sigma', mu=1.16, sigma=0.34, shape=(1, npeaks))

        if baseline == 'offset':
            a0 = pm.Uniform('a0', 0, max_amp, shape=(len(observations), 1))
            # f(x) = sum of gaussian peaks + offset
            y = pm.Deterministic('y', (amp.T * np.exp(-(xvalues - mu.T) ** 2 / (2 * sigma.T ** 2))).sum(axis=0) + a0)
        elif baseline == 'linear':
            a0 = pm.Uniform('a0', 0, max_amp, shape=(len(observations), 1))
            a1 = pm.Uniform('a1', 0, max_amp/(xvalues.max()-xvalues.min()))
            # f(x) = sum of gaussian peaks + offset
            y = pm.Deterministic('y', (amp.T * np.exp(-(xvalues - mu.T) ** 2 / (2 * sigma.T ** 2))).sum(axis=0) 
                + a0 + a1 * xvalues)
        else:
            # f(x) = sum of gaussian peaks
            y = pm.Deterministic('y', (amp.T * np.exp(-(xvalues - mu.T) ** 2 / (2 * sigma.T ** 2))).sum(axis=0))

        # noise prior
        sigma_e = pm.Gamma('sigma_e', alpha=1., beta=1.)
        epsilon = pm.HalfNormal('epsilon', sd=sigma_e)

        # likelihood
        y_pred = pm.Normal('y_pred', mu=y, sd=epsilon, observed=observations)

        return model
