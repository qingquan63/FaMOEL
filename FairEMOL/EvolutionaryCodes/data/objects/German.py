from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np

class German(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'german'
        self.class_attr = 'Probability'
        self.positive_class_val = 1
        self.negative_class_val = 0
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_class_names = {'sex': 1, 'age': 1}
        self.categorical_features = []
        self.features_to_keep = ['credit_history', 'savings', 'employment', 'sex', 'age', 'Probability']
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        dataset_orig = dataframe
        dataset_orig = dataset_orig.dropna()

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])

        # mean = dataset_orig.loc[:,"age"].mean()
        # dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1,
                                                  dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1,
                                                  dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1,
                                                  dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2,
                                                  dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3,
                                                  dataset_orig['credit_history'])

        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])

        ## ADD Columns
        dataset_orig['credit_history=Delay'] = 0
        dataset_orig['credit_history=None/Paid'] = 0
        dataset_orig['credit_history=Other'] = 0

        dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1,
                                                        dataset_orig['credit_history=Delay'])
        dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1,
                                                            dataset_orig['credit_history=None/Paid'])
        dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1,
                                                        dataset_orig['credit_history=Other'])

        dataset_orig['savings_more_than_500'] = 0
        dataset_orig['savings_less_than_500'] = 0
        dataset_orig['savings_Unknown'] = 0

        dataset_orig['savings_more_than_500'] = np.where(dataset_orig['savings'] == 1, 1,
                                                         dataset_orig['savings_more_than_500'])
        dataset_orig['savings_less_than_500'] = np.where(dataset_orig['savings'] == 2, 1,
                                                         dataset_orig['savings_less_than_500'])
        dataset_orig['savings_Unknown'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings_Unknown'])

        dataset_orig['employment=1-4 years'] = 0
        dataset_orig['employment=4+ years'] = 0
        dataset_orig['employment=Unemployed'] = 0

        dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1,
                                                        dataset_orig['employment=1-4 years'])
        dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1,
                                                       dataset_orig['employment=4+ years'])
        dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1,
                                                         dataset_orig['employment=Unemployed'])

        dataset_orig = dataset_orig.drop(['credit_history', 'savings', 'employment'], axis=1)
        ## In dataset 1 means good, 2 means bad for probability. I change 2 to 0
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)


        return dataset_orig
