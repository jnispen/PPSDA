import matplotlib.pyplot as plt
import numpy as np

def plot_datasets(ldata, lpeaks, dims, figure_size=(12,16), *args, **kwargs):
    """ plots a list of datasets and optionally saves the figure """
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)

    # subplot dimensions
    nrows = dims[0]
    ncols = dims[1]

    _, ax = plt.subplots(nrows, ncols, figsize=figure_size, constrained_layout=True)
    ax = np.ravel(ax)
    for idx, data in enumerate(ldata):
        x_val = np.array(data.columns.to_list(), dtype='float32')
        X = data.columns
        Y = data[X].values
        #mu = lpeaks[idx]
        mu = np.array(lpeaks[idx], dtype=float)
        for i in range(len(data)):
            #ax[idx].plot(X, Y[i], "-", alpha=.5)
            ax[idx].plot(x_val, Y[i], "-", alpha=.5)
        for j in range(len(mu)):
            ax[idx].axvline(mu[j], linestyle='--', color='gray', alpha=.5)
        ax[idx].set_title("#{}".format(idx +1))

    if savefig == 'yes':
        plt.savefig(fname + '.png', dpi=150)

def plot_posterior(x_val, data_val, traces, ppc_traces, dims, figure_size=(12,16), *args, **kwargs):
    """ plots the posterior of a list of traces and optionally saves the figure """
    savefig = kwargs.get('savefig', None)
    fname = kwargs.get('fname', None)
    showpeaks = kwargs.get('showpeaks', None)

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
            y_val = data_val[idx].values[i]
            ax[idx].plot(x_val, y_val, '-', color="red", alpha=.2, linewidth=1)
        ax[idx].set_title("#{}".format(idx + 1))

    if savefig == 'yes' and showpeaks == 'yes':
        plt.savefig(fname + '_ppc_peaks.png', dpi=150)
    elif savefig == 'yes': 
        plt.savefig(fname + '_ppc.png', dpi=150)
