import pandas as pd
import numpy as np

""" Class representing the data object """
class Data:
    """ Used to load and store data """
    def __init__(self, filename, label_column='label', non_data_columns=0):
        """ initialise class, load data file, shuffle data rows"""
        self.label_column=label_column
        self.non_data_columns=non_data_columns
        self.data = pd.read_csv(filename)
        self.data = self.data.sample(frac=1, random_state=42)
        self.classes = len(self.get_class_labels())

    def get_class_labels(self):
        """ return list of class labels """
        #class_labels = list({lbl for lbl in self.data.iloc[:,
        #                    self.data.columns.to_list().index(self.label_column)].tolist()})
        #class_labels.sort()
        return np.unique(self.data.iloc[:,self.data.columns.to_list().index(self.label_column)].tolist())

    def get_x_val(self):
        """ return ndarray of x values """
        cols = self.data.columns.to_list()
        return np.array(cols[:self.non_data_columns], dtype='float32')
