import numpy as np
import pandas as pd
import csv

def data_generator(xvalues, nsamples=15, npeaks=3, peakshape=0, noise=0.05, scatter='no', tbaseline='none'):
    """ dataset generator for pseudo-Voigt profiles """
    """ parameters:
        xvalues   = list of xvalues (wavelengths)
        nsamples  = number of samples to generate (observations, default 15 samples)
        npeaks    = number of peaks in simulated spectrum (3 default)
        peakshape = peakshape parametr id pseudoVoigt formula (0 = Gauss, 1 = Lorentz)
        noise     = instrumental noise level (% of the minimal signal amplitude, 1% default)
        scatter   = light scatter constant (0.8x to 1.2x of peak amplitude, default no)
        baseline_type = type of baseline in the data (none/offset/linear/quadratic, default no baseline)
         
        returns: pandas dataframe containing the simulated spectra
                 list containing the peak positions
                 pandas dataframe containing peak information (peak position and peak amplitude)  
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

    return (pd.DataFrame(data=Y, columns=X), mu, pd.DataFrame({'mu': mu, 'amp': amp}))

def add_peakshift(peaklist, peak_shift=0.0):
    """ adds a shift to the supplied peaklist """
    lpeaks_shifted = []
    for i, val in enumerate(peaklist):
        ps = np.array(val, dtype=float) + peak_shift
        lpeaks_shifted += [ps]
    return lpeaks_shifted

def data_load(count, path):
    """ load generated datasets and peak information from disk """
    """ parameters:
        count = number of datasets to load (0-99)
        path  = directory to load datasets from

        returns: list of pandas dataframes containing the simulated spectra
                 list containing the peak positions 
    """
    # read datasets
    ldata = []
    for i in range(count):
        df = pd.read_csv(path + '/dataset_%02d.csv' % (i + 1))
        ldata.append(df)
    # read peak information
    with open(path + '/peakinfo.csv', newline='') as fp:
        reader = csv.reader(fp)
        lpeaks = list(reader)

    return (ldata, lpeaks)

def data_save(filename, peaklist):
    """ save a list of peak maxima to .csv file """
    """ parameters
        filename = file to save
        peaklist = list containing peak maxima
    """
    with open(filename, mode='w') as fp:
        fwr = csv.writer(fp, delimiter=',')
        for i, val in enumerate(peaklist):
            fwr.writerow(val)
