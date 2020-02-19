import pandas as pd
import arviz as az

def get_results_summary(varnames, traces, ppc_traces, y_values):
    """ function to collect summary statistics from a list of traces and models """
    # -- sampling statistics --
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

    #  -- model statistics --
    waic = [az.waic(traces[i]).waic for i in range(len(traces))]

    # r2 scores
    r2 = []
    for idx, ppc_x in enumerate(ppc_traces):
        score = az.r2_score(y_values[idx], ppc_x['y_pred'])
        r2.append(score.r2)

    # build dataframe and return
    df = pd.DataFrame()
    df['r_hat'] = r_hat
    df['mcse'] = mc_err
    df['ess'] = ess
    df['bfmi'] = bfmi
    df['r2'] = r2
    df['waic'] = waic
    df['epsilon'] = epsilon
    df.index += 1

    return df
