from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'Probability'
        self.positive_class_val = '>50K'
        self.negative_class_val = '<=50K'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = {'race': 'White', 'sex': 'Male'}
        self.categorical_features = []
        self.features_to_keep = ['age', 'education-num', 'race', 'sex', 'capital-gain',
                                 'capital-loss', 'hours-per-week', 'Probability']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe



