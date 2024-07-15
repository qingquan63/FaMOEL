from FairEMOL.EvolutionaryCodes.data.objects.Data import Data

class Bank(Data):
    # 参考https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/bank_dataset.py
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'bank'
        self.class_attr = 'Probability'
        self.positive_class_val = 'yes'
        self.negative_class_val = 'no'

        self.sensitive_attrs = ['age']
        self.privileged_class_names = {'age': 'adult'}
        self.categorical_features = ['default', 'housing', 'loan']
        self.features_to_keep = ['age', 'default', 'balance', 'housing', 'day', 'loan', 'duration',
                                 'campaign', 'pdays', 'previous', 'Probability']
        self.missing_val_indicators = ["unknown"]

    def data_specific_processing(self, dataframe):
        old = dataframe['age'] >= 30
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        return dataframe
