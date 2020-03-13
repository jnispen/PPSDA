import matplotlib.pyplot as plt
import numpy as np

def plot_datasets(ldata, lpeaks, dims, figure_size=(12,16), *args, **kwargs):
    """ plots a list of datasets and optionally saves the figure """
    
    # optional arguments
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    title = kwargs.get('title', None)
    bline = kwargs.get('baselines', None)

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
        if title == 'peaks':
            ax[idx].set_title("#{0} ({1}p)".format(idx+1,len(lpeaks[idx])))
        if title == 'baseline':
            ax[idx].set_title("#{0} (baseline = {1})".format(idx+1,bline[idx]))
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
    # labels containing model/peak combination (used in scenario c)
    labels = kwargs.get('labels', None)
    
    if labels is not None:
        lmodel = [i[0] for i in labels]
        lpeak  = [i[1] for i in labels]
        
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
            ax[idx].set_title("#{0} ({1}-peak model:{2}-peak spectrum)"
              .format(idx+1,lmodel[idx],lpeak[idx]))
        else:
            ax[idx].set_title("#{0}".format(idx+1))

    if savefig == 'yes' and showpeaks == 'yes':
        plt.savefig(fname + '_ppc_peaks.png', dpi=150)
    elif savefig == 'yes': 
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
    
    # labels containing model/peak combination (used in scenario c)
    label = kwargs.get('labels', None)

    plt.figure(figsize=figure_size)
    
    if posteriors is not None:
        # plot samples from the posterior
        sp = posteriors['y_pred']
        for i in range(15):
            plt.plot(x_val, sp[-i, i, :], '-', color="black", alpha=.2)

    if priors is not None:
        # plot samples from the prior
        sp = priors['y_pred']
        for i in range(15):
            plt.plot(x_val, sp[-i, i, :], '--', color="blue", alpha=.2)

    # plot samples from Y (peak number = idx)
    for i in range(len(traces)):
        A = traces['amp'][i].flatten()
        M = traces['mu'][i].flatten()
        S = traces['sigma'][i].flatten()
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
    if label is not None:
        plt.title("({0}-peak model:{1}-peak spectrum)"
              .format(label[0],label[1]))
            
    if savefig == 'yes' and showpeaks == 'yes':
        plt.savefig(fname + '_ppc_peaks.png', dpi=150)
    elif savefig == 'yes': 
        plt.savefig(fname + '_ppc.png', dpi=150)

    