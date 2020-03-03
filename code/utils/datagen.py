import numpy as np
import pandas as pd

def data_generator(xvalues, nsamples=15, npeaks=3, peakshape=0, noise=0.1, scatter='no', tbaseline='none'):
    """ dataset generator for pseudo-Voigt profiles """
    """ parameters:
        xvalues   = list of xvalues (wavelengths)
        nsamples  = number of samples to generate (observations)
        npeaks    = number of peaks in simulated spectrum
        peakshape = peakshape parametr id pseudoVoigt formula (0 = Gauss, 1 = Lorentz)
        epsilon   = instumental noise level 
        scatter   = light scatter constant
        baseline_type = type of baseline in the data (none/offset/linear/quadratic)
         
        returns: pandas dataframe containing the simulated spectra
                 list containing the peak positions  
    """

    # number of features (wavelengths)
    xsize = len(xvalues)

    # convert xvalues to numpy array
    X = np.array(xvalues)

    # graph scale parameters
    amp_min = 5
    amp_max = 25
    xdiff = X.max()-X.min()

    # generate list of peak parameters
    amp   = np.random.uniform(low=amp_min, high=amp_max, size=npeaks)
    mu    = np.random.uniform(low=X.min(), high=X.max(), size=npeaks)
    sigma = np.random.lognormal(mean=1.16, sigma=0.34, size=npeaks)

    # noise level
    epsilon = noise

    # numpy array containing individual peaks
    profiles = np.zeros((npeaks, xsize))
    for i in range(npeaks):
        profiles[i,:] = np.array(amp[i] * (peakshape * (sigma[i]**2/((X - mu[i])**2 + sigma[i]**2)) +
                                   (1-peakshape) * np.exp(-(X - mu[i])**2/(2*sigma[i]**2)) ))

    # numpy array containing individual baselines
    baselines = np.zeros((nsamples, xsize))
    for i in range(nsamples):
        if tbaseline == 'offset':
            a0 = np.random.uniform(low=amp_min, high=amp_max)
            baselines[i,:] = np.array(a0)
        elif tbaseline == 'linear':
            a0 = np.random.uniform(low=amp_min, high=amp_max)
            a1 = np.random.uniform(low=0, high=amp_max/xdiff)
            baselines[i, :] = np.array(a0 + a1 * X)
        elif tbaseline == 'quadratic':
            a0 = np.random.uniform(low=amp_min, high=amp_max)
            a1 = np.random.uniform(low=0, high=amp_max/xdiff)
            a2 = np.random.uniform(low=0, high=amp_max/(xdiff**2))
            baselines[i, :] = np.array(a0 + a1 * X + a2 * X**2)
        else:
            baselines[i, :] = np.array(0)

    # add peaks, baseline and noise into cumulative sample
    Y = np.zeros((nsamples, xsize))
    for i in range(nsamples):
        if scatter == 'yes':
            cscat = np.random.uniform(low=0.8, high=1.2)
            Y[i,:] = cscat * profiles.sum(axis=0) + baselines[i] + np.random.randn(xsize) * epsilon
        else:
            Y[i,:] = profiles.sum(axis=0) + baselines[i] + np.random.randn(xsize) * epsilon

    return (pd.DataFrame(data=Y, columns=X), mu)

def data_load(count, path, shift_factor=1.0):
    """ load generated datasets and peak information from disk """
    """ parameters:
        count = number of datasets to load (0-99)
        path  = directory to load datasets from
        
        returns: list of pandas dataframes containing the simulated spectra
                 list containing the peak positions 
    """

    ldata = []
    for i in range(count):
        df = pd.read_csv(path + '/dataset_%02d.csv' % (i+1))
        ldata.append(df)
    lpeaks = np.loadtxt(path + '/peakinfo.csv', delimiter=',')

    # shift peak information by factor
    lpeaks = lpeaks * shift_factor

    return (ldata, lpeaks)