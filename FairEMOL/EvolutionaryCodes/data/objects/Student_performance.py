from FairEMOL.EvolutionaryCodes.data.objects.Data import Data



class Student_performance(Data):
    # categorical_features 中没有sensitive attrs 和 class_attr
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'student_performance'
        self.class_attr = 'Probability'
        self.positive_class_val = 'Excellent'
        self.negative_class_val = 'Other'
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = {'sex': 'M'}
        self.categorical_features = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                                     'romantic']
        self.features_to_keep = ['sex', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'schoolsup',
                                 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                                 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'Probability']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        good = dataframe['Probability'] > 12
        dataframe.loc[good, 'Probability'] = 'Excellent'
        nogood = dataframe['Probability'] != 'Excellent'
        dataframe.loc[nogood, 'Probability'] = 'Other'
        return dataframe



