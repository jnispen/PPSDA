import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_color(key):
    """ return a color for class """
    color_dict = {0: "blue", 1: "red", 2: "green", 3: "orange", 4: "black"}
    return color_dict[key]

def plot_data(data, non_data_columns, label_column='label', nrows=50):
    """ plot the dataframe """
    # header data = x-values
    cols = data.columns.to_list()
    x_val = np.array(cols[:non_data_columns], dtype='float32')

    # plot rows (== observations) in a single figure
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # used to map label codes to colors
    label_codes = pd.Categorical(data[label_column]).codes

    # list of class labels
    clabels = list({lbl for lbl in data.iloc[:, data.columns.to_list().index(label_column)].tolist()})
    clabels.sort()
    for i in range(len(clabels)):
        print(str(clabels[i]) + ": " + get_color(i))

    for i in range(nrows):
        y_val = data.values[i]
        plt.plot(x_val, y_val[:non_data_columns], '-', color=get_color(label_codes[i]))

def plot_real_vs_ppc(data, ppc_class_lst, non_data_columns, class_labels, label_column='label', nrows=10):
    """ plot real data vs. posterior samples """
    # header data = x-values
    cols = data.columns.to_list()
    x_val = np.array(cols[:non_data_columns], dtype='float32')

    plt.figure(figsize=(12, 8))
    plt.axes()

    # plot some samples from the posterior
    for i in range(5):
        for z in range(len(ppc_class_lst)):
            plt.plot(x_val, ppc_class_lst[z][i, 0, :], 'o-', color="gray", alpha=.3)

    # plot mean data for classes (raw data)
    df = []
    for i in range(len(class_labels)):
        df.append(data.loc[data[label_column] == class_labels[i]].sample(frac=1))

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
