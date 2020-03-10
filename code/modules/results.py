import pandas as pd
import arviz as az
import numpy as np

def get_results_summary(varnames, traces, ppc_traces, y_values, *args, **kwargs):
    """ function to collect summary statistics from a list of traces and models """

    # noise level (used in scenario a)
    noise_level = kwargs.get('epsilon_real', None)
    # run number (used in multiple runs scenario)
    run_number  = kwargs.get('runlist', None)
    # total number of datasets per series (used in scenario c)
    tsets = kwargs.get('sets', None)
    # labels containing model/peak combination (used in scenario c)
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

def get_model_summary(data, peaklist):
    """ function to extract convergence information from a dataframe """
    """ parameters:
            data     : list of dataframes containing convergence results 
                       (see get_results_summary) 
            peaklist : list with peak numbers
            
            returns  : dictionary containing NxN matrices with the average 
                       convergence results (per model/data combination)
    """
    
    npeaks = len(peaklist)
    count = 0
    
    waic_mat  = np.full((npeaks,npeaks),0.0)
    rhat_mat  = np.full((npeaks,npeaks),0.0)
    r2_mat    = np.full((npeaks,npeaks),0.0)
    bfmi_mat  = np.full((npeaks,npeaks),0.0)
    mcse_mat  = np.full((npeaks,npeaks),0.0)
    noise_mat = np.full((npeaks,npeaks),0.0)
    ess_mat   = np.full((npeaks,npeaks),0.0)

    # loop over all the dataframes in the datalist
    for idx, dat in enumerate(data):
        print("processing dataframe: ", idx+1)
        print("number of runs      : ", dat['run'].max())
        # loop over all runs in the dataframe
        # select the runs 1-by-1
        for k in range(dat['run'].max()):
            df = dat.loc[(dat['run'] == (k+1))]
            #print("select run          : ", k+1)
            count += 1           
            # loop over models and number of peaks and average
            # the results per model/peaknumber combination
            for i, val in enumerate(peaklist):
                sel1 = df.loc[(df['model'] == val)]
                for j, val in enumerate(peaklist):
                    sel2 = sel1.loc[(sel1['peaks'] == val)]
                    
                    waic_mat[i][j]  += sel2['waic'].mean()
                    rhat_mat[i][j]  += sel2['r_hat'].mean()
                    r2_mat[i][j]    += sel2['r2'].mean()
                    bfmi_mat[i][j]  += sel2['bfmi'].mean()
                    mcse_mat[i][j]  += sel2['mcse'].mean()
                    noise_mat[i][j] += sel2['epsilon'].mean()
                    ess_mat[i][j]   += sel2['ess'].mean()
    
    return {'waic' : waic_mat/count, 
            'rhat' : rhat_mat/count,
            'r2'   : r2_mat/count,
            'bfmi' : bfmi_mat/count,
            'mcse' : mcse_mat/count, 
            'noise': noise_mat/count, 
            'ess'  : ess_mat/count}
