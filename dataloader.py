import numpy as np
import pandas as pd
from scipy.special import expit
from Utils import Utils

class DataLoader:
    def load_train_test_data_random(self, csv_path, split_size=0.8):
        print("Loading data...")
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, no = \
            self.preprocess_dataset_for_training(csv_path)
            
        print("Covariate shape: {0}".format(np_covariates_X.shape))
        print("Treatment shape: {0}".format(np_treatment_T.shape))

        # Random split
        idx = np.random.permutation(no)
        train_idx = idx[:int(split_size * no)]
        test_idx = idx[int(split_size * no):]

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, split_size)

        print("Train Statistics:")
        print(f"Features shape: {np_train_X.shape}")
        print(f"Treatment shape: {np_train_T.shape}")
        n_treated = np_train_T[np_train_T == 1].shape[0]
        n_total = np_train_T.shape[0]

        print("Test Statistics:")
        print(f"Features shape: {np_test_X.shape}")
        print(f"Treatment shape: {np_test_T.shape}")

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    @staticmethod
    def preprocess_dataset_for_training(csv_path):
        # Read data
        data = pd.read_csv(csv_path)
        
        # Process categorical variables
        # One-hot encode gender_cd
        gender_dummies = pd.get_dummies(data['gender_cd'], prefix='gender')
        
        # Normalize numerical features
        numerical_features = ['age', 'hpd_hyp', 'hpd_hyc', 'hpd_ast', 'hpd_dia']
        numerical_data = data[numerical_features]
        
        # Min-max normalization for numerical features
        normalized_numerical = (numerical_data - numerical_data.min()) / (numerical_data.max() - numerical_data.min())
        
        # Combine normalized numerical features with one-hot encoded categorical features
        feature_matrix = pd.concat([normalized_numerical, gender_dummies], axis=1)
        
        # Convert to numpy array
        x = feature_matrix.values
        no, dim = x.shape

        # Get treatment and outcome
        t = data['grp_binary'].values
        y = data['social_risk_score'].values
        
        # Normalize outcome
        y = (y - y.min()) / (y.max() - y.min())

        # Convert to numpy arrays and reshape
        t = Utils.convert_to_col_vector(t)
        y = Utils.convert_to_col_vector(y)

        # Create factual and counterfactual outcomes
        y_f = np.where(t == 1, y, 0)  # Factual outcomes
        y_cf = np.where(t == 0, y, 0)  # Counterfactual outcomes
        
        y_f = Utils.convert_to_col_vector(y_f)
        y_cf = Utils.convert_to_col_vector(y_cf)
        
        print("Feature dimensions after one-hot encoding:", dim)
        print("Processed features:", feature_matrix.columns.tolist())

        return x, t, y_f, y_cf, no

    @staticmethod
    def get_feature_dimensions(csv_path):
        """
        Calculate the total input dimensions after one-hot encoding
        """
        data = pd.read_csv(csv_path)
        gender_unique = data['gender_cd'].nunique()
        numerical_features = 5  # age, hpd_hyp, hpd_hyc, hpd_ast, hpd_dia
        total_dimensions = numerical_features + gender_unique
        return total_dimensions