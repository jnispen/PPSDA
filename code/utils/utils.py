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

def plot_data(cdata, nrows=50):
    """ plot the dataframe """
    # header data = x-values
    x_val = cdata.get_x_val()

    # plot rows (== observations) in a single figure
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # used to map label codes to colors
    label_codes = pd.Categorical(cdata.data[cdata.label_column]).codes

    # list of class labels
    clabels = cdata.get_class_labels()

    for i in range(len(clabels)):
        print(str(clabels[i]) + ": " + get_color(i))

    for i in range(nrows):
        y_val = cdata.data.values[i]
        plt.plot(x_val, y_val[:cdata.non_data_columns], '-', color=get_color(label_codes[i]))

def plot_mean_vs_ppc(cdata, ppc_class_lst):
    """ plot data mean vs. posterior samples """
    # header data = x-values
    x_val = cdata.get_x_val()

    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # plot a sample from the posterior (for each class)
    for i in range(1):
        for z in range(len(ppc_class_lst)):
           plt.plot(x_val, ppc_class_lst[z][i, 0, :], '-', color=get_color(z), alpha=.6)

    # list of class labels
    class_labels = cdata.get_class_labels()

    # plot the posterior mean
    for z in range(len(ppc_class_lst)):
        cls_label = str(class_labels[z]) + " ppc mean"
        plt.plot(x_val, ppc_class_lst[z][:, 0].mean(axis=0), '-', color=get_color(z), alpha=.6, label=cls_label)

    # plot mean data for classes (raw data)
    df = [ cdata.data.loc[cdata.data[cdata.label_column] == class_labels[k]]
           for k in range(len(class_labels)) ]
    for z in range(len(df)):
        cls_label = str(class_labels[z]) + " real mean"
        plt.plot(x_val, df[z].iloc[:,:cdata.non_data_columns].mean(), '--', color=get_color_mean(z),
             label=cls_label, linewidth=1)

    # plot 94% HPD interval
    for z in range(len(ppc_class_lst)):
        col = "C" + str(z+1)
        az.plot_hpd(x_val, ppc_class_lst[z], smooth=False, color=col)

    plt.legend(loc='best')

def plot_real_vs_ppc(cdata,  ppc_class_lst, nrows=10):
    """ plot real data vs. posterior samples """
    # header data = x-values
    x_val = cdata.get_x_val()

    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')

    # plot some samples from the posterior
    for i in range(5):
        for z in range(len(ppc_class_lst)):
            plt.plot(x_val, ppc_class_lst[z][i, 0, :], 'o-', color="gray", alpha=.3)

    # list of class labels
    class_labels = cdata.get_class_labels()

    # plot raw data for classes
    df = [ cdata.data.loc[cdata.data[cdata.label_column] == class_labels[i]].sample(frac=1)
           for i in range(len(class_labels)) ]
    for i in range(nrows):
        for z in range(len(df)):
            plt.plot(x_val, df[z].values[i,:cdata.non_data_columns], '--', color=get_color(z), linewidth=1)

def append_predictions(cdata, trace, test_data, display=True):
    """ appends predicted labels to the test dataframe """
    # check model predictions on test dataset
    a = trace['alpha'].mean()
    b = trace['beta'].mean(axis=0)

    xt_n = test_data.columns[:cdata.non_data_columns]
    xt_s = test_data[xt_n].values
    xt_s = (xt_s - xt_s.mean(axis=0)) / xt_s.std(axis=0)

    mu_t = a + (b * xt_s).sum(axis=1)
    yt_p = 1 / (1 + np.exp(-mu_t))

    pt_y = np.zeros(len(xt_s))
    lp_t = []

    class_labels = cdata.get_class_labels()

    for i in range(len(xt_s)):
        if yt_p[i] < 0.5:
            pt_y[i] = 0
            lp_t.append(class_labels[0])
        else:
            pt_y[i] = 1
            lp_t.append(class_labels[1])

    #test_data = test_data.assign(pred=pd.Series(pt_y))
    test_data = test_data.assign(p_label=pd.Series(lp_t))

    if display:
        print(test_data.iloc[:, (cdata.non_data_columns-1):])

    return test_data

def append_predictions_ppc(cdata, trace, display=True):
    """ appends predicted labels to the dataframe """
    # check model predictions on test dataset
    a = trace['alpha'].mean()
    b = trace['beta'].mean(axis=0)

    xt_n = cdata.data.columns[:cdata.non_data_columns]
    xt_s = cdata.data[xt_n].values
    xt_s = (xt_s - xt_s.mean(axis=0)) / xt_s.std(axis=0)

    mu_t = a + (b * xt_s).sum(axis=1)
    yt_p = 1 / (1 + np.exp(-mu_t))

    pt_y = np.zeros(len(xt_s))
    lp_t = []

    class_labels = cdata.get_class_labels()

    for i in range(len(xt_s)):
        if yt_p[i] < 0.5:
            pt_y[i] = 0
            lp_t.append(class_labels[0])
        else:
            pt_y[i] = 1
            lp_t.append(class_labels[1])

    #cdata.data = cdata.data.assign(pred=pd.Series(pt_y))
    cdata.data = cdata.data.assign(p_label=pd.Series(lp_t))

    if display:
        print (cdata.data.iloc[:,(cdata.non_data_columns-1):])

def get_score(data, label_column, predicted_column):
    """ calculates the logreg score for a single column """
    yt = pd.Categorical(data[label_column]).codes
    cor = 0; err = 0
    for i in range(len(yt)):
        if data[label_column][i] == data[predicted_column][i]:
            cor += 1
        else:
            err += 1

    tot = len(yt)
    score = f'{cor / len(yt) * 100:.1f}'

    return tot, cor, err, score

def logistic_score(data, label_column, predicted_column, kfold=False):
    """ calculates and prints the logistic score """
    if kfold:
        print('    tot  cor  err  score')
        print('----------------------------')
        ttot = 0; tcor = 0; terr = 0
        for i in range(len(data)):
            tot, cor, err, score = get_score(data[i], label_column, predicted_column)
            print(str(i) + "    " + str(tot) + "   " + str(cor) +  "   " + str(err) + "   " + score + "%")
            ttot += tot; tcor += cor; terr += err
        print('----------------------------')
        print("     " + str(ttot) + "   " + str(tcor) + "   " + str(terr) + "   " + f'{tcor / ttot * 100:.1f}' + "%")
    else:
        tot, cor, err, score = get_score(data, label_column, predicted_column)
        print("total  : " + str(tot))
        print("correct: " + str(cor))
        print("error  : " + str(err))
        print("score  : " + score + "%")

def logistic_score_ppc(cdata, predicted_column):
    """ calculates and prints the logistic regression score """
    yt = pd.Categorical(cdata.data[cdata.label_column]).codes
    cor = 0; err = 0
    for i in range(len(yt)):
        if cdata.data[cdata.label_column][i] == cdata.data[predicted_column][i]:
            cor += 1
        else:
            err += 1

    print("total  : " + str(len(yt)))
    print("correct: " + str(cor))
    print("error  : " + str(err))
    print("score  : " + f'{cor / len(yt) * 100:.1f}' + "%")

def softmax_score(cdata, data, trace):
    """ calculates and prints the softmax score """
    yt = pd.Categorical(data[cdata.label_column]).codes
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

def save_traces(cdata, filename, samples_per_class, ppc_class_lst):
    """ saves the trace to a .csv file """
    import csv

    # create header row
    file_header = cdata.get_x_val()
    header = np.array(np.around(file_header, 3), dtype='str')
    header = header.tolist()
    header.append("label")

    # list of class labels
    class_labels = cdata.get_class_labels()

    with open(filename, mode='w') as fp:
        ppc_writer = csv.writer(fp, delimiter=',')
        ppc_writer.writerow(header)

        for i in range(samples_per_class):
            for z in range(len(ppc_class_lst)):
                row = np.array(ppc_class_lst[z][i, 0, :], dtype='str').tolist()
                row.append(class_labels[z])
                ppc_writer.writerow(row)

def standardize(x):
    """ standardizes the data X, substracts the mean and normalizes the variance """
    return (x - x.mean(axis=0)) / x.std(axis=0)

def train_test_split_kfold(data, nfolds):
    """ splits the dataframe into n-fold sets of training and test data """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=nfolds, random_state=42)
    kf.get_n_splits(data)

    # list of training and test dataframes
    train_lst, test_lst = [], []

    for train_index, test_index in kf.split(data):
        # create empty dataframe
        df_train = pd.DataFrame(columns=data.columns.to_list())
        df_test = pd.DataFrame(columns=data.columns.to_list())

        # add rows to training and test dataframes, re-index and append to list
        for i, val in enumerate(train_index):
            df_train.loc[data.index[val]] = data.iloc[val]
        df_train.index = range(len(df_train))
        train_lst.append(df_train)

        for i, val in enumerate(test_index):
            df_test.loc[data.index[val]] = data.iloc[val]
        df_test.index = range(len(df_test))
        test_lst.append(df_test)

    return train_lst, test_lst
