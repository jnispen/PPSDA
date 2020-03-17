import pandas as pd
import arviz as az
import numpy as np

# local module imports
import models as mdl

def get_results_summary(traces, ppc_traces, y_values, varnames=[], *args, **kwargs):
    """ function to collect summary statistics from a list of traces and models """

    # noise level (used in scenario a)
    noise_level = kwargs.get('epsilon_real', None)
    # run number (used in multiple runs scenario)
    run_number  = kwargs.get('runlist', None)
    # total number of datasets per series (used in scenario b,c)
    tsets = kwargs.get('sets', None)
    # labels containing model/peak (used in scenario c)
    labels = kwargs.get('labels', None)
    # with multiple models, each model might have different variables (yes/no)
    multi_models = kwargs.get('multimodels', None)
    
    #####
    # statistics on the sampling
    ####
    r_hat = []
    ess = []
    mc_err = []
    epsilon = []
    for idx, trace in enumerate(traces):
        if multi_models == 'yes':
            varnames = mdl.get_varnames(trace)
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
        df['data'] = [i[1] for i in labels]
    df.index += 1

    return df

def load_results(filelist):
    """ loads results from a list of .csv files """
    """ parameters:
           filelist : list of files to load
    
           returns  : list of dataframes containing the results
    """
    ldf = []
    for idx, val in enumerate(filelist):
        print("reading file: {0}".format(val))
        df = pd.read_csv(val)
        df.index += 1
        ldf += [df]
        
    return ldf

def add_means(ddict, sel_data, lblist):
    """ add means of selected items in the dictionary """
    """ parameters:
           ddict    : dictionary containing NxN data matrices
           sel_data : dataframe containing values to add to dictionary values
           lblist   : list containing labels with peak values, baselines, etc.
    """
    
    # loop over models and number of peaks and average
    # the results per model/peaknumber combination
    for i, val in enumerate(lblist):
        sel1 = sel_data.loc[(sel_data['model'] == val)]
        for j, val in enumerate(lblist):
            sel2 = sel1.loc[(sel1['data'] == val)]

            ddict['waic'][i][j]  += sel2['waic'].mean()
            ddict['rhat'][i][j]  += sel2['r_hat'].mean()
            ddict['r2'][i][j]    += sel2['r2'].mean()
            ddict['bfmi'][i][j]  += sel2['bfmi'].mean()
            ddict['mcse'][i][j]  += sel2['mcse'].mean()
            ddict['noise'][i][j] += sel2['epsilon'].mean()
            ddict['ess'][i][j]   += sel2['ess'].mean()
                
    return ddict
                
def get_model_summary(data, labellist, *args, **kwargs):
    """ function to extract convergence information from a dataframe """
    """ parameters:
            data      : list of dataframes containing convergence results 
                        (see get_results_summary) 
            labellist : list containing labels with peak values, baselines, etc.
            
            returns   : dictionary containing NxN matrices with the average 
                        convergence results (per model/data combination)
    """
    
    len_labels = len(labellist)
    count = 0
    
    waic_mat  = np.full((len_labels,len_labels),0.0)
    rhat_mat  = np.full((len_labels,len_labels),0.0)
    r2_mat    = np.full((len_labels,len_labels),0.0)
    bfmi_mat  = np.full((len_labels,len_labels),0.0)
    mcse_mat  = np.full((len_labels,len_labels),0.0)
    noise_mat = np.full((len_labels,len_labels),0.0)
    ess_mat   = np.full((len_labels,len_labels),0.0)

    # dictionary containing convergence information
    cdict = {'waic' : waic_mat, 
             'rhat' : rhat_mat,
             'r2'   : r2_mat,
             'bfmi' : bfmi_mat,
             'mcse' : mcse_mat, 
             'noise': noise_mat, 
             'ess'  : ess_mat}

    if len(data) > 1:
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
                cdict = add_means(cdict, df, labellist)
    else:
        count = 1
        cdict = add_means(cdict, data[0], labellist)
        
    # calculate the average
    for key in cdict:
        cdict[key] /= count 

    return cdict