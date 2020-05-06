import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import seaborn as sns

def plot_datasets(ldata, lpeaks, dims, figure_size=(12,16), *args, **kwargs):
    """ plots a list of datasets and optionally saves the figure """
    
    # optional arguments
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    scenario = kwargs.get('scenario', None)
    labels = kwargs.get('labels', None)
    title = kwargs.get('title', None)
    
    # subplot dimensions
    nrows = dims[0]
    ncols = dims[1]

    _, ax = plt.subplots(nrows, ncols, figsize=figure_size, constrained_layout=True)
    ax = np.ravel(ax)
    for idx, data in enumerate(ldata):
        x_val = np.array(data.columns.to_list(), dtype='float32')
        X = data.columns
        Y = data[X].values
        mu = np.array(lpeaks[idx], dtype=float)
        for i in range(len(data)):
            ax[idx].plot(x_val, Y[i], "-", alpha=.5)
        for j in range(len(mu)):
            ax[idx].axvline(mu[j], linestyle='--', color='gray', alpha=.5)
        if scenario == 'peaks':
            ax[idx].set_title("#{0} ({1}p)".format(idx+1,len(lpeaks[idx])))
        elif scenario == 'baseline':
            ax[idx].set_title("#{0} (baseline = {1})".format(idx+1,labels[idx]))
        elif scenario == 'peakshape':
            ax[idx].set_title("#{0} (n = {1})".format(idx+1,labels[idx]))
        elif scenario == 'single':
            ax[idx].set_title(title)
        else:
            ax[idx].set_title("#{0}".format(idx+1))

    if savefig == 'yes':
        plt.savefig(fname + '.png', dpi=150)

def plot_posterior(x_val, data_val, traces, ppc_traces, dims, figure_size=(12,16), *args, **kwargs):
    """ plots the posterior of a list of traces and optionally saves the figure """
    
    # optional arguments
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    showpeaks = kwargs.get('showpeaks', None)
    tsets = kwargs.get('sets', None)
    scenario = kwargs.get('scenario', None)
    # labels containing model/data combination (used in scenario b/c)
    labels = kwargs.get('labels', None)
    
    if labels is not None:
        lcola = [i[0] for i in labels]
        lcolb = [i[1] for i in labels]
        
    # subplot dimensions
    nrows = dims[0]
    ncols = dims[1]

    _, ax = plt.subplots(nrows, ncols, figsize=figure_size, constrained_layout=True)
    ax = np.ravel(ax)
    for idx, ppc_x in enumerate(ppc_traces):
        # plot samples from the posterior
        sp = ppc_x['y_pred']
        for i in range(10):
            ax[idx].plot(x_val, sp[-i, i, :], '-', color="black", alpha=.2)

        # plot samples from the prior
        #sp = ppc_prior['y_pred']
        #for i in range(10):
        #    ax[i].plot(x_val, sp[-i, i, :], '-', color="blue", alpha=.2)

        if showpeaks == 'yes':
            # plot mixture components
            A = traces[idx]['amp'].mean(axis=0).flatten()
            M = traces[idx]['mu'].mean(axis=0).flatten()
            S = traces[idx]['sigma'].mean(axis=0).flatten()
            for j in range(len(A)):
                Y = A[j] * np.exp(-(x_val - M[j]) ** 2 / (2 * S[j] ** 2))
                ax[idx].plot(x_val, Y, '--', linewidth=1)
                ax[idx].axvline(M[j], linestyle='--', linewidth=1, color='g')
                ax[idx].errorbar(x=M[j], y=.5 * A[j], xerr=S[j], fmt='o',
                                 ecolor='r', elinewidth=1, capsize=5, capthick=1)

        # plot samples from the dataset
        for i in range(5):
            # use modulo indexing for multiple model plotting
            if tsets is not None:
                index = idx % tsets
            else:
                index = idx
            y_val = data_val[index].values[i]
            ax[idx].plot(x_val, y_val, '-', color="red", alpha=.2, linewidth=1)
        if labels is not None:
            if scenario == 'peaks':
                ax[idx].set_title("#{0} ({1}-peak model:{2}-peak data)"
                  .format(idx+1,lcola[idx],lcolb[idx]))
            if scenario == 'baseline':
                ax[idx].set_title("#{0} ({1} model:{2} data)"
                  .format(idx+1,lcola[idx],lcolb[idx]))
            if scenario == 'peakshape':
                ax[idx].set_title("#{0} (n={1} model:n={2} data)"
                  .format(idx+1,lcola[idx],lcolb[idx]))
        else:
            ax[idx].set_title("#{0}".format(idx+1))

    if savefig == 'yes' and showpeaks == 'yes':
        plt.savefig(fname + '_ppc_peaks.png', dpi=150)
    elif savefig == 'yes': 
        plt.savefig(fname + '_ppc.png', dpi=150)

def plot_heatmap(data, labellist, title, color, fsize, fname="./heatmap", precision=".3f"):
    ''' plots a heatmap from numerical data provided in a NxN matrix '''
    
    sns.set(font_scale=1.3)
    
    
    plt.figure(figsize=fsize, tight_layout=True)
    plt.title(title)

    yticks = ["m_{0}".format(str(val)) for _, val in enumerate(labellist)]
    xticks = ["d_{0}".format(str(val)) for _, val in enumerate(labellist)]
    
    #sns.heatmap(data, annot=True, fmt=precision, linewidths=1, linecolor="#efefef", square=True,
    #                cmap=color, cbar=False, xticklabels=xticks, yticklabels=yticks)
    sns.heatmap(data, annot=True, fmt=precision, linewidths=1, square=True,
                    cmap=color, cbar=False, xticklabels=xticks, yticklabels=yticks)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.savefig(fname + '.png', dpi=150)

def plot_posterior_n(x_val, data_val, lpeaks, traces, ppc_traces, figure_size=(12,16), *args, **kwargs):
    """ plots the posterior of a single trace and optionally saves the figure """
    
    # optional arguments
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    title = kwargs.get('title', None)
    
    plt.figure(figsize=figure_size, constrained_layout=True)

    for idx, ppc_x in enumerate(ppc_traces):
        # plot samples from the posterior
        sp = ppc_x['y_pred']
        for i in range(10):
            plt.plot(x_val, sp[-i, i, :], '-', color="black", alpha=.2)

        # plot 94% HPD interval
        az.plot_hpd(x_val, ppc_x['y_pred'], smooth=False, credible_interval=0.95, color='#FFFF00')

        # plot samples from the dataset
        #for i in range(5):
            # use modulo indexing for multiple model plotting
        #    index = idx
        #    y_val = data_val[index].values[i]
        #    plt.plot(x_val, y_val, '-', color="red", alpha=.2, linewidth=1)
        mu = np.array(lpeaks[idx], dtype=float)
        for j in range(len(mu)):
            plt.axvline(mu[j], linestyle='--', color='gray', alpha=.5)
        plt.title(title)

    if savefig == 'yes': 
        plt.savefig(fname + '_ppc.png', dpi=150)
    
def plot_posterior_single(x_val, data_val, traces, figure_size=(12,16), *args, **kwargs):
    """ plots the posterior of a single trace and optionally saves the figure """
    
    # optional arguments
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    showpeaks = kwargs.get('showpeaks', None)
    posteriors = kwargs.get('posteriors', None)
    priors = kwargs.get('priors', None)
    idx = kwargs.get('peakidx', None)
    samples = kwargs.get('samples', None)
    scenario = kwargs.get('scenario', None)
    
    # labels containing model/data combination (used in scenario b/c)
    label = kwargs.get('labels', None)

    plt.figure(figsize=figure_size)
    
    if posteriors is not None:
        # plot samples from the posterior
        sp = posteriors['y_pred']
        a = np.arange(0,len(sp),len(sp)/10, dtype=int)
        for i in a:
            plt.plot(x_val, sp[i, 0, :], '-', color="black", alpha=.2)
            
        # plot 94% HPD interval
        az.plot_hpd(x_val, posteriors['y_pred'], smooth=False, color= 'C1')

    if priors is not None:
        # plot samples from the prior
        sp = priors['y_pred']
        for i in range(15):
            plt.plot(x_val, sp[-i, i, :], '--', color="blue", alpha=.2)   
    
    # plot samples from Y (peak number = idx)
    l = len(traces['mu'])
    print("len trace:", l)
    a = np.arange(0,l,l/50, dtype=int)
    print("len subsample:", len(a))
    for i in a:
        A = traces['amp'][i].flatten()
        M = traces['mu'][i].flatten()
        S = traces['sigma'][i].flatten()
        #print("M[{0}]: {1}".format(i,M[idx]))
        Y = A[idx] * np.exp(-(x_val - M[idx]) ** 2 / (2 * S[idx] ** 2))
        plt.plot(x_val, Y, '-', linewidth=1, color="green", alpha=.5)

    if showpeaks == 'yes':
        # plot mixture components
        A = traces['amp'].mean(axis=0).flatten()
        M = traces['mu'].mean(axis=0).flatten()
        S = traces['sigma'].mean(axis=0).flatten()
        for j in range(len(A)):
            Y = A[j] * np.exp(-(x_val - M[j]) ** 2 / (2 * S[j] ** 2))
            plt.plot(x_val, Y, '--', linewidth=1)
            plt.axvline(M[j], linestyle='--', linewidth=1, color='g')
            plt.errorbar(x=M[j], y=.5 * A[j], xerr=S[j], fmt='o',
                               ecolor='r', elinewidth=1, capsize=5, capthick=1)

    if samples == 'yes':
        # plot samples from the dataset
        for i in range(10):
            y_val = data_val.values[i]
            plt.plot(x_val, y_val, '-', color="red", alpha=.2, linewidth=1)
    if scenario == 'peaks':
        plt.title("({0}-peak model:{1}-peak data)"
              .format(label[0],label[1]))
            
    if savefig == 'yes' and showpeaks == 'yes':
        plt.savefig(fname + '_ppc_peaks.png', dpi=150)
    elif savefig == 'yes': 
        plt.savefig(fname + '_ppc.png', dpi=150)
