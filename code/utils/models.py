import pymc3 as pm

def generative_model(observations, nclasses, nfeatures, *args, **kwargs):
    """ basic generative model """
    nsamples = kwargs.get('nsamples', None)

    with pm.Model() as model_gm:
        # priors
        mu_m = pm.Lognormal('mu_m', mu=0, sigma=1)
        sigma_ab = pm.Gamma('sigma_ab', alpha=1., beta=1.)
        sigma_mu = pm.HalfNormal('sigma_mu', sigma_ab)
        mu = pm.Normal('mu', mu=mu_m, sd=sigma_mu, shape=(nclasses, nfeatures))

        sigma_s = pm.Gamma('sigma_s', alpha=1., beta=1.)
        sigma = pm.HalfNormal('sigma', sigma_s)

        # likelihood
        if nsamples != None:
            y_pred = [pm.Normal('class_%d' % i, mu=mu[i], sd=sigma,
                        observed=observations[i][:nsamples]) for i in range(nclasses)]
        else:
            y_pred = [pm.Normal('class_%d' % i, mu=mu[i], sd=sigma,
                        observed=observations[i][:len(observations[i])]) for i in range(nclasses)]

        return model_gm
