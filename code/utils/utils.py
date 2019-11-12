import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

    # map label codes to colors
    label_codes = pd.Categorical(data[label_column]).codes
    color_dict = {0: "blue", 1: "red", 2: "green", 3: "orange", 4: "black"}

    for i in range(nrows):
        y_val = data.values[i]
        plt.plot(x_val, y_val[:non_data_columns], '-', color=color_dict[label_codes[i]])

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
