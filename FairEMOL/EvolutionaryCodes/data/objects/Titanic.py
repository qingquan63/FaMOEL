from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np


class Titanic(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'Titanic'
        self.class_attr = 'Probability'
        self.positive_class_val = 1  # survived
        self.negative_class_val = 0
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = {'sex': 'male'}
        self.categorical_features = []
        self.features_to_keep = ['Pclass', 'sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Probability']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe

