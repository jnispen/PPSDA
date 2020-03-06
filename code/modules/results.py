import pandas as pd
import arviz as az

def get_results_summary(varnames, traces, ppc_traces, y_values, *args, **kwargs):
    """ function to collect summary statistics from a list of traces and models """

    # noise level (used in scenario a)
    noise_level = kwargs.get('epsilon_real', None)
    # run number (used in multiple runs scenario)
    run_number  = kwargs.get('runlist', None)
    # total number of datasets per series (used in scenario c)
    tsets = kwargs.get('sets', None)
    # labels for inference run (used in scenario c)
    labels = kwargs.get('labels', None)

    #####
    # statistics on the sampling
    ####
    r_hat = []
    ess = []
    mc_err = []
    epsilon = []
    for idx, trace in enumerate(traces):
        coef = az.summary(trace, varnames)
        r_hat.append(coef['r_hat'].mean())
        ess.append(coef['ess_mean'].mean())
        mc_err.append(coef['mcse_mean'].mean())
        epsilon.append(trace['epsilon'].mean())
    # BFMI sampling data
    bfmi = [az.bfmi(traces[i]).mean() for i in range(len(traces))]

    #####
    #  statistics on the model
    ####
    waic = [az.waic(traces[i]).waic for i in range(len(traces))]

    # r2 scores
    r2 = []
    for idx, ppc_x in enumerate(ppc_traces):
        if tsets is not None:
            # use modulo indexing for multiple model calculations
            index = idx % tsets
        else:
            index = idx
        score = az.r2_score(y_values[index], ppc_x['y_pred'])
        r2.append(score.r2)

    # build dataframe and return results
    df = pd.DataFrame()
    df['r_hat'] = r_hat
    df['mcse'] = mc_err
    df['ess'] = ess
    df['bfmi'] = bfmi
    df['r2'] = r2
    df['waic'] = waic
    df['epsilon'] = epsilon
    if noise_level is not None:
        df['epsilon_real'] = noise_level
    if run_number is not None:
        df['run'] = run_number
    if labels is not None:
        df['model'] = [i[0] for i in labels]
        df['peaks'] = [i[1] for i in labels]
    df.index += 1

    return df
