from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np

class Heart_health(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'heart_health'
        self.class_attr = 'Probability'
        self.positive_class_val = 1
        self.negative_class_val = 0
        self.sensitive_attrs = ['age']
        self.privileged_class_names = {'age': 'young'}
        self.categorical_features = []
        self.features_to_keep = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                 'oldpeak', 'slope', 'ca', 'thal', 'Probability']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        mean_age = np.mean(dataframe['age'])
        old = dataframe['age'] >= mean_age
        dataframe.loc[old, 'age'] = 'old'
        young = dataframe['age'] != 'old'
        dataframe.loc[young, 'age'] = 'young'

        p = dataframe['Probability'] > 0
        dataframe.loc[p, 'Probability'] = 1
        n_p = dataframe['Probability'] != 1
        dataframe.loc[n_p, 'Probability'] = 0
        return dataframe



