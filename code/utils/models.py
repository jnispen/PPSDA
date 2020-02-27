import numpy as np
import pymc3 as pm

def generative_model(observations, nclasses, nfeatures, *args, **kwargs):
    """ basic generative model """
    nsamples = kwargs.get('nsamples', None)

    with pm.Model() as model:
        # priors
        mu_m = pm.Lognormal('mu_m', mu=0, sigma=1)
        sigma_ab = pm.Gamma('sigma_ab', alpha=1., beta=1.)
        sigma_mu = pm.HalfNormal('sigma_mu', sigma_ab)
        mu = pm.Normal('mu', mu=mu_m, sd=sigma_mu, shape=(nclasses, nfeatures))

        sigma_s = pm.Gamma('sigma_s', alpha=1., beta=1.)
        sigma = pm.HalfNormal('sigma', sigma_s)

        # likelihood
        if nsamples is not None:
            y_pred = [pm.Normal('class_%d' % i, mu=mu[i], sd=sigma,
                        observed=observations[i][:nsamples]) for i in range(nclasses)]
        else:
            y_pred = [pm.Normal('class_%d' % i, mu=mu[i], sd=sigma,
                        observed=observations[i][:len(observations[i])]) for i in range(nclasses)]

        return model

def get_max(observations_lst):
    max_val = float("-inf")
    for i in range(len(observations_lst)):
        row_max = observations_lst[i].max()
        if row_max > max_val:
            max_val = row_max
    return max_val

def model_gauss(observations, xvalues, npeaks, *args, **kwargs):
    """ basic model: the spectrum is assumed as gaussian peaks + noise """
    mu_peaks = kwargs.get('mu_peaks', None)

    # maximum peak amplitude
    max_amp = get_max(observations)

    with pm.Model() as model:
        # priors for Gaussian peak shapes
        amp = pm.Uniform('amp', 0, max_amp, shape=(1, npeaks))

        if mu_peaks is not None:
            #print("mu_peaks: ", mu_peaks)
            #mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
            #               shape=(1, npeaks), transform=pm.distributions.transforms.ordered, testval=mu_peaks)
            # LogNormal model
            mu = pm.Uniform('mu', xvalues.min(), xvalues.max(), shape=(1, npeaks), testval=mu_peaks)
        else:
            #mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
            #               shape=(1, npeaks), transform=pm.distributions.transforms.ordered)
            # LogNormal model
            mu = pm.Uniform('mu', xvalues.min(), xvalues.max(), shape=(1, npeaks))

        # Normal model
        #sigma = pm.HalfNormal('sigma', sd=100, shape=(1, npeaks))
        # LogNormal model
        sigma = pm.Lognormal('sigma', mu=1.16, sigma=0.34, shape=(1, npeaks))

        # f(x) = gaussian peaks
        y = pm.Deterministic('y', (amp.T * np.exp(-(xvalues - mu.T) ** 2 / (2 * sigma.T ** 2))).sum(axis=0))

        # noise prior
        sigma_e = pm.Gamma('sigma_e', alpha=1., beta=1.)
        epsilon = pm.HalfNormal('epsilon', sd=sigma_e)

        # likelihood
        y_pred = pm.Normal('y_pred', mu=y, sd=epsilon, observed=observations)

        return model

def model_gauss_constant(observations, nclasses, xvalues, npeaks, *args, **kwargs):
    """ basic gaussian peak model + constant y-offset """
    nsamples = kwargs.get('nsamples', None)
    mu_peaks = kwargs.get('mu_peaks', None)

    # maximum peak amplitude
    max_amp = get_max(observations)

    with pm.Model() as model:
        # priors for Gaussian peak shape
        amp = pm.Uniform('amp', 0, max_amp, shape=(nclasses, npeaks))

        if mu_peaks is not None:
            mu = pm.Normal('mu', mu=mu_peaks, sd=50,
                       shape=(nclasses, npeaks), transform=pm.distributions.transforms.ordered)
        else:
            mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
                       shape=(nclasses, npeaks), transform=pm.distributions.transforms.ordered)

        sigma = pm.HalfNormal('sigma', sd=100, shape=(nclasses, npeaks))

        # priors for constant y-offset
        sigma_aa = pm.Gamma('sigma_aa', alpha=1., beta=1.)
        sigma_a = pm.HalfNormal('sigma_a', sd=sigma_aa)

        if nsamples is not None:
            a_ = [pm.Normal('a_%d' % i, mu=0, sd=sigma_a, shape=(nsamples, 1)) for i in range(nclasses)]
            #a_ = [pm.Normal('a_%d' % i, mu=0, sd=sigma_a) for i in range(nclasses)]
        else:
            a_ = [pm.Normal('a_%d' % i, mu=0, sd=sigma_a, shape=(len(observations[i]), 1)) for i in range(nclasses)]
            #a_ = [pm.Normal('a_%d' % i, mu=0, sd=sigma_a) for i in range(nclasses)]

        #print("amp  : ", amp.tag.test_value.shape)
        #print("mu   : ", mu.tag.test_value.shape)
        #print("sigma: ", sigma.tag.test_value.shape)
        #for i in range(nclasses):
        #    print("a_%d  : " % i, a_[i].tag.test_value.shape)
        #print("xvalues: ", xvalues.shape)

        # f(x) = gaussian peaks + constant baseline
        y_ = [pm.Deterministic('y_%d' % i, (amp[i] * np.exp(-(xvalues - mu[i]) ** 2 / (2 * sigma[i] ** 2))).sum(axis=1)
                               + a_[i]) for i in range(nclasses)]

        #for i in range(nclasses):
        #    print("y_%d  : " % i, y_[i].tag.test_value.shape)

        # noise prior
        sigma_e = pm.Gamma('sigma_e', alpha=1., beta=1.)
        epsilon = pm.HalfNormal('epsilon', sd=sigma_e)

        # likelihood
        if nsamples is not None:
            y_pred = [pm.Normal('class_%d' % i, mu=y_[i], sd=epsilon, observed=observations[i][:nsamples])
                      for i in range(nclasses)]
        else:
            y_pred = [pm.Normal('class_%d' % i, mu=y_[i], sd=epsilon, observed=observations[i][:len(observations[i])])
                      for i in range(nclasses)]

        return model

def model_gauss_linear(observations, nclasses, xvalues, npeaks, *args, **kwargs):
    """ basic gaussian peak model + constant y-offset """
    nsamples = kwargs.get('nsamples', None)
    mu_peaks = kwargs.get('mu_peaks', None)

    # maximum peak amplitude
    max_amp = get_max(observations)

    with pm.Model() as model:
        # priors for Gaussian peak shape
        amp = pm.Uniform('amp', 0, max_amp, shape=(nclasses, npeaks))

        if mu_peaks is not None:
            mu = pm.Normal('mu', mu=mu_peaks, sd=50,
                       shape=(nclasses, npeaks), transform=pm.distributions.transforms.ordered)
        else:
            mu = pm.Normal('mu', mu=np.linspace(xvalues.min(), xvalues.max(), npeaks), sd=50,
                       shape=(nclasses, npeaks), transform=pm.distributions.transforms.ordered)

        sigma = pm.HalfNormal('sigma', sd=100, shape=(nclasses, npeaks))

        # priors for constant y-offset
        sigma_aa = pm.Gamma('sigma_aa', alpha=1., beta=1.)
        sigma_a = pm.HalfNormal('sigma_a', sd=sigma_aa)

        if nsamples is not None:
            a0_ = [pm.Normal('a0_%d' % i, mu=0, sd=sigma_a, shape=(nsamples, 1)) for i in range(nclasses)]
            a1_ = [pm.Normal('a1_%d' % i, mu=0, sd=sigma_a, shape=(nsamples, 1)) for i in range(nclasses)]
        else:
            a0_ = [pm.Normal('a0_%d' % i, mu=0, sd=sigma_a, shape=(len(observations[i]), 1)) for i in range(nclasses)]
            a1_ = [pm.Normal('a1_%d' % i, mu=0, sd=sigma_a, shape=(len(observations[i]), 1)) for i in range(nclasses)]

        #print("amp  : ", amp.tag.test_value.shape)
        #print("mu   : ", mu.tag.test_value.shape)
        #print("sigma: ", sigma.tag.test_value.shape)
        for i in range(nclasses):
            print("a0_%d  : " % i, a0_[i].tag.test_value.shape)
        print("xvalues: ", xvalues.shape)

        # f(x) = gaussian peaks + linear baseline
        y_ = [pm.Deterministic('y_%d' % i, (amp[i] * np.exp(-(xvalues - mu[i]) ** 2 / (2 * sigma[i] ** 2))).sum(axis=1)
                               + a0_[i] + (a1_[i] * xvalues.T).sum(axis=0)) for i in range(nclasses)]

        for i in range(nclasses):
            print("y_%d  : " % i, y_[i].tag.test_value.shape)

        # noise prior
        sigma_e = pm.Gamma('sigma_e', alpha=1., beta=1.)
        epsilon = pm.HalfNormal('epsilon', sd=sigma_e)

        # likelihood
        if nsamples is not None:
            y_pred = [pm.Normal('class_%d' % i, mu=y_[i], sd=epsilon, observed=observations[i][:nsamples])
                      for i in range(nclasses)]
        else:
            y_pred = [pm.Normal('class_%d' % i, mu=y_[i], sd=epsilon, observed=observations[i][:len(observations[i])])
                      for i in range(nclasses)]

        return model
