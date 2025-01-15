import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import expit

from Utils import Utils

class DataLoader:
    def load_train_test_lalonde_random(self, csv_path, split_size=0.8):
        # Data preprocessing
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, no \
            = self.preprocess_dataset_for_training(csv_path)
            
        print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_treatment_T.shape))

        idx = np.random.permutation(no)
        train_idx = idx[:int(split_size * no)]
        test_idx = idx[int(split_size * no):]

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, split_size)

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)
        n_treated = np_train_T[np_train_T == 1]
        
        n_treated = n_treated.shape[0]
        n_total = np_train_T.shape[0]

        print("Numpy test Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    @staticmethod
    def preprocess_dataset_for_training(csv_path):
        # Read data
        df = pd.read_csv(csv_path)
        
        # Create dummy variables for race
        df = pd.get_dummies(df, columns=['race'])
        
        # Select features
        feature_cols = ['age', 'married', 'educ', 'nodegree', 're74', 're75'] + [col for col in df.columns if col.startswith('race_')]
        
        # Standardize continuous variables
        scaler = StandardScaler()
        continuous_cols = ['age', 'educ', 're74', 're75']
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
        
        # Prepare X, T, Y
        x = df[feature_cols].values
        x = x.astype(np.float32)  # Convert to float32
        no, dim = x.shape
        
        # Treatment assignment
        t = df['treat'].values
        t = Utils.convert_to_col_vector(t)
        
        # Observed outcome (re78)
        y_f = df['re78'].values
        
        # Normalize outcomes to [0,1] using MinMaxScaler
        minmax_scaler = MinMaxScaler()
        y_f = minmax_scaler.fit_transform(y_f.reshape(-1, 1))
        y_f = y_f.astype(np.float32)  # Convert to float32
        
        # Generate synthetic counterfactuals using a simple linear model
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1]).astype(np.float32)
        z = np.matmul(x, coef)
        noise = np.random.normal(0, 0.01, size=[no, 1]).astype(np.float32)
        prob_temp = expit(z + noise)
        
        # Estimate counterfactual outcomes
        y_cf = np.zeros_like(y_f, dtype=np.float32)
        treated_idx = (t == 1).squeeze()
        control_idx = (t == 0).squeeze()
        
        # Simple estimation of counterfactuals based on observed outcomes
        y_cf[treated_idx] = np.mean(y_f[control_idx]) + \
                           (y_f[treated_idx] - np.mean(y_f[treated_idx]))
        y_cf[control_idx] = np.mean(y_f[treated_idx]) + \
                           (y_f[control_idx] - np.mean(y_f[control_idx]))
        
        # Ensure counterfactuals are also in [0,1]
        y_cf = np.clip(y_cf, 0, 1)
        
        return x, t, y_f, y_cf, no