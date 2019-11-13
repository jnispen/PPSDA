import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def get_color(key):
    """ return color for class key"""
    color_dict = {0: "blue", 1: "red", 2: "green", 3: "orange", 4: "black"}
    return color_dict[key]

def get_color_mean(key):
    """ return mean color for class key"""
    color_dict = {0: "yellow", 1: "black", 2: "green", 3: "orange", 4: "white"}
    return color_dict[key]

def get_data_x_value_header(data, non_data_columns):
    """ returns a ndarray of the data x values """
    cols = data.columns.to_list()
    return np.array(cols[:non_data_columns], dtype='float32')

def get_class_labels(data, label_column):
    """ returns a sorted list of class labels """
    class_labels = list({lbl for lbl in data.iloc[:, data.columns.to_list().index(label_column)].tolist()})
    class_labels.sort()
    return class_labels

def plot_data(data, non_data_columns, label_column='label', nrows=50):
    """ plot the dataframe """
    # header data = x-values
    x_val = get_data_x_value_header(data, non_data_columns)

    # plot rows (== observations) in a single figure
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # used to map label codes to colors
    label_codes = pd.Categorical(data[label_column]).codes

    # list of class labels
    clabels = get_class_labels(data, label_column)

    for i in range(len(clabels)):
        print(str(clabels[i]) + ": " + get_color(i))

    for i in range(nrows):
        y_val = data.values[i]
        plt.plot(x_val, y_val[:non_data_columns], '-', color=get_color(label_codes[i]))

def plot_mean_vs_ppc(data, ppc_class_lst, non_data_columns, label_column='label'):
    """ plot data mean vs. posterior samples """
    # header data = x-values
    x_val = get_data_x_value_header(data, non_data_columns)

    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # plot a sample from the posterior (for each class)
    for i in range(1):
        for z in range(len(ppc_class_lst)):
           plt.plot(x_val, ppc_class_lst[z][i, 0, :], '-', color=get_color(z), alpha=.6)

    # list of class labels
    class_labels = get_class_labels(data, label_column)

    # plot the posterior mean
    for z in range(len(ppc_class_lst)):
        cls_label = str(class_labels[z]) + " ppc mean"
        plt.plot(x_val, ppc_class_lst[z][:, 0].mean(axis=0), '-', color=get_color(z), alpha=.6, label=cls_label)

    # plot mean data for classes (raw data)
    df = [ data.loc[data[label_column] == class_labels[k]] for k in range(len(class_labels)) ]
    for z in range(len(df)):
        cls_label = str(class_labels[z]) + " real mean"
        plt.plot(x_val, df[z].iloc[:,:non_data_columns].mean(), '--', color=get_color_mean(z),
             label=cls_label, linewidth=1)

    # plot 94% HPD interval
    for z in range(len(ppc_class_lst)):
        col = "C" + str(z+1)
        az.plot_hpd(x_val, ppc_class_lst[z], smooth=False, color=col)

    plt.legend(loc='best')

def plot_real_vs_ppc(data,  ppc_class_lst, non_data_columns, label_column='label', nrows=10):
    """ plot real data vs. posterior samples """
    # header data = x-values
    x_val = get_data_x_value_header(data, non_data_columns)

    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # plot some samples from the posterior
    for i in range(5):
        for z in range(len(ppc_class_lst)):
            plt.plot(x_val, ppc_class_lst[z][i, 0, :], 'o-', color="gray", alpha=.3)

    # list of class labels
    class_labels = get_class_labels(data, label_column)

    # plot raw data for classes
    df = [ data.loc[data[label_column] == class_labels[i]].sample(frac=1) for i in range(len(class_labels)) ]
    for i in range(nrows):
        for z in range(len(df)):
            plt.plot(x_val, df[z].values[i,:non_data_columns], '--', color=get_color(z), linewidth=1)

def display_predictions(trace, test_data, non_data_columns, class_labels):
    """ displays predicted labels next to the real labels """
    # check model predictions on test dataset
    a = trace['alpha'].mean()
    b = trace['beta'].mean(axis=0)

    xt_n = test_data.columns[:non_data_columns]
    xt_s = test_data[xt_n].values
    xt_s = (xt_s - xt_s.mean(axis=0)) / xt_s.std(axis=0)

    mu_t = a + (b * xt_s).sum(axis=1)
    yt_p = 1 / (1 + np.exp(-mu_t))

    pt_y = np.zeros(len(xt_s))
    lp_t = []

    for i in range(len(xt_s)):
        if yt_p[i] < 0.5:
            pt_y[i] = 0
            lp_t.append(class_labels[0])
        else:
            pt_y[i] = 1
            lp_t.append(class_labels[1])

    test_data = test_data.assign(pred=pd.Series(pt_y))
    test_data = test_data.assign(pred_label=pd.Series(lp_t))

    print(test_data.iloc[:,(non_data_columns-2):])

def logistic_score(data, label_column, predicted_labels):
    """ calculates and prints the logistic score """
    yt = pd.Categorical(data[label_column]).codes
    cor = 0; err = 0
    for i in range(len(yt)):
        if data[label_column].iloc[i] == predicted_labels[i]:
            cor += 1
        else:
            err += 1

    print("total  : " + str(len(yt)))
    print("correct: " + str(cor))
    print("error  : " + str(err))
    print("score  : " + f'{cor / len(yt) * 100:.1f}' + "%")

def softmax_score(data, trace, label_column):
    """ calculates and prints the softmax score """
    yt = pd.Categorical(data[label_column]).codes
    data_pred = trace['mu'].mean(0)
    y_pred = [np.exp(point) / np.sum(np.exp(point), axis=0)
              for point in data_pred]
    cor = 0; err = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == yt[i]:
            cor += 1
        else:
            err +=1

    print("total  : " + str(len(y_pred)))
    print("correct: " + str(cor))
    print("error  : " + str(err))
    print("score  : " + f'{cor / len(y_pred) * 100:.1f}' + "%")

def standardize(x):
    """ standardizes the data X, substracts the mean and normalizes the variance """
    return (x - x.mean(axis=0)) / x.std(axis=0)

def save_traces(filename, samples_per_class, file_header, class_labels, ppc_class_lst):
    """ saves the trace to a .csv file """
    import csv

    # create header row
    header = np.array(np.around(file_header, 3), dtype='str')
    header = header.tolist()
    header.append("label")

    with open(filename, mode='w') as fp:
        ppc_writer = csv.writer(fp, delimiter=',')
        ppc_writer.writerow(header)

        for i in range(samples_per_class):
            for z in range(len(ppc_class_lst)):
                row = np.array(ppc_class_lst[z][i, 0, :], dtype='str').tolist()
                row.append(class_labels[z])
                ppc_writer.writerow(row)
