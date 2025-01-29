import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from Utils import Utils

class DataLoader:
    def __init__(self):
        self.outcome_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        # Models for implementing SITA
        self.ps_model = LogisticRegression(random_state=42)
        self.outcome_models = {
            0: GradientBoostingRegressor(random_state=42),  # For control group
            1: GradientBoostingRegressor(random_state=42)   # For treated group
        }

    def load_train_test_lalonde_random(self, csv_path, split_size=0.8):
        # Data preprocessing
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, no \
            = self.preprocess_dataset_for_training(csv_path)
            
        print("LaLonde Dataset Statistics:")
        print(f"Features shape: {np_covariates_X.shape}")
        print(f"Treatment shape: {np_treatment_T.shape}")

        # Create train/test split indices
        idx = np.random.permutation(no)
        train_idx = idx[:int(split_size * no)]
        test_idx = idx[int(split_size * no):]

        # Split the data
        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, split_size)

        # Calculate dataset statistics
        n_treated = np.sum(np_train_T == 1)
        n_total = len(np_train_T)
        
        print("\nTraining Set Statistics:")
        print(f"Total samples: {n_total}")
        print(f"Treated samples: {n_treated} ({n_treated/n_total:.3f})")
        print(f"Control samples: {n_total - n_treated} ({1 - n_treated/n_total:.3f})")

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    def preprocess_dataset_for_training(self, csv_path):
        # Read and preprocess data
        df = pd.read_csv(csv_path)
        
        # Create dummy variables for race
        df = pd.get_dummies(df, columns=['race'])
        
        # Select features
        feature_cols = ['age', 'married', 'educ', 'nodegree', 're74', 're75'] + \
                      [col for col in df.columns if col.startswith('race_')]
        
        # Scale features
        continuous_cols = ['age', 'educ', 're74', 're75']
        df[continuous_cols] = self.feature_scaler.fit_transform(df[continuous_cols])
        
        # Prepare feature matrix
        X = df[feature_cols].values.astype(np.float32)
        T = df['treat'].values.reshape(-1, 1).astype(np.float32)
        
        # Process outcome (re78)
        Y = df['re78'].values.reshape(-1, 1)
        Y_scaled = self.outcome_scaler.fit_transform(Y).astype(np.float32)
        
        # Fit propensity score model
        self.ps_model.fit(X, T.ravel())
        ps_scores = self.ps_model.predict_proba(X)[:, 1]
        
        # Fit outcome models for treatment and control
        treated_idx = (T == 1).squeeze()
        control_idx = (T == 0).squeeze()
        
        self.outcome_models[1].fit(X[treated_idx], Y_scaled[treated_idx])
        self.outcome_models[0].fit(X[control_idx], Y_scaled[control_idx])
        
        # Estimate potential outcomes
        Y_0 = self.outcome_models[0].predict(X).reshape(-1, 1)
        Y_1 = self.outcome_models[1].predict(X).reshape(-1, 1)
        
        # Assign factual and counterfactual outcomes
        Y_f = Y_scaled
        Y_cf = np.where(T == 1, Y_0, Y_1)
        
        return X, T, Y_f, Y_cf, len(X)

    def estimate_individual_outcomes(self, X):
        """
        Estimate potential outcomes using trained models
        Returns E[Y|T=0, X=x] and E[Y|T=1, X=x]
        """
        Y_0 = self.outcome_models[0].predict(X).reshape(-1, 1)
        Y_1 = self.outcome_models[1].predict(X).reshape(-1, 1)
        return Y_0, Y_1

    def calculate_effects(self, X, T):
        """
        Calculate treatment effects in normalized scale
        """
        Y_0, Y_1 = self.estimate_individual_outcomes(X)
        
        # Individual Treatment Effect
        ite = Y_1 - Y_0
        
        # Average Treatment Effect
        ate = np.mean(ite)
        
        return ite, ate

    def inverse_transform_outcomes(self, outcomes):
        """Transform outcomes back to original dollar scale"""
        if isinstance(outcomes, np.ndarray):
            outcomes = outcomes.reshape(-1, 1)
        else:
            outcomes = np.array(outcomes).reshape(-1, 1)
        return self.outcome_scaler.inverse_transform(outcomes).flatten()

    def get_propensity_scores(self, X):
        """Get propensity scores π(x) = P(T=1|X=x)"""
        return self.ps_model.predict_proba(X)[:, 1]
