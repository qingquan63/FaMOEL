from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np


class PropublicaRecidivism(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-recidivism'
        self.class_attr = 'two_year_recid'
        self.positive_class_val = 1
        self.negative_class_val = 0
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_class_names = {'sex': 'Male', 'race': 'Caucasian'}
        self.categorical_features = ['age_cat', 'c_charge_degree', 'priors_count', 'score_text']
        self.features_to_keep = ["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score",
                                 "score_text", "two_year_recid", "days_b_screening_arrest"]

        ## Drop NULL values
        self.missing_val_indicators = []
        # self.dele_data = {'race': {'Asian', 'Native American', 'Hispanic', 'Other'}}

    def data_specific_processing(self, dataframe):
        dataset_orig = dataframe

        # dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Male', 1, 0)
        # dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
        dataset_orig['priors_count'] = np.where(
            (dataset_orig['priors_count'] >= 1) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
        dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45', 45, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
        dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

        # dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

        return dataset_orig
