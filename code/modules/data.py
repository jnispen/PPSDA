import pandas as pd
import numpy as np

""" class representing the data object """
class Data:
    
    def __init__(self, filename, label_column='label', *args, **kwargs):
        """ data class initialisation """   
        # optional arguments
        shuffle = kwargs.get('shuffle', None)

        # column name containing the class labels
        self.label_column = label_column

        # read dataset from .csv file
        self.data = pd.read_csv(filename)
        
        # number of classes in dataset
        self.classes = len(self.get_class_labels())

        # optionally shuffle the data
        if shuffle == 'yes':
            self.data = self.data.sample(frac=1)

    def get_class_labels(self):
        """ return list of class labels """
        #class_labels = list({lbl for lbl in self.data.iloc[:,
        #                    self.data.columns.to_list().index(self.label_column)].tolist()})
        #class_labels.sort()
        return np.unique(self.data.iloc[:,self.data.columns.to_list().index(self.label_column)].tolist())
