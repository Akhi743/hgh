import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataLoader:
    def __init__(self):
        self.y_min = None
        self.y_max = None
        self.preprocessor = None
        
    def data_loading_custom(self, train_data_split_rate=0.8):
        """Load custom dataset for GANITE with proper preprocessing.
        
        Args:
            train_data_split_rate: proportion of training data
            
        Returns:
            train_x: features in training data
            train_t: treatments in training data
            train_y: observed outcomes in training data
            train_potential_y: potential outcomes in training data
            test_x: features in testing data
            test_potential_y: potential outcomes in testing data
        """
        # Load the data
        data = pd.read_csv("subset_500_rows.csv")
        
        # Define numeric and categorical columns
        numeric_features = ['age', 'hpd_hyp', 'hpd_hyc', 'hpd_ast', 'hpd_dia']
        categorical_features = ['gender_cd']
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop=None, sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Transform features
        x = self.preprocessor.fit_transform(data[numeric_features + categorical_features])
        
        # Get feature names
        numeric_feature_names = numeric_features
        categorical_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_feature_names + list(categorical_feature_names)
        
        no, dim = x.shape
        print(f"Number of samples: {no}")
        print(f"Number of features after one-hot encoding: {dim}")
        print(f"Feature names: {all_feature_names}")
        
        # Treatment assignment
        t = data['grp_binary'].values.astype(np.float32)
        
        # Outcome
        y = data['social_risk_score'].values.astype(np.float32)
        
        # Store original scale for rescaling
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        print(f"Original scale - Min: {self.y_min:.4f}, Max: {self.y_max:.4f}")
        
        # Normalize outcome to [0,1] range
        y = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Train/test split
        idx = np.random.permutation(no)
        train_idx = idx[:int(train_data_split_rate * no)]
        test_idx = idx[int(train_data_split_rate * no):]
        
        train_x = x[train_idx, :]
        train_t = t[train_idx]
        train_y = y[train_idx]
        
        test_x = x[test_idx, :]
        test_t = t[test_idx]
        test_y = y[test_idx]
        
        # Generate potential outcomes
        train_potential_y = self._generate_potential_outcomes(train_x, train_t, train_y)
        test_potential_y = self._generate_potential_outcomes(test_x, test_t, test_y)
        
        return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y

    def _generate_potential_outcomes(self, x, t, y, noise_std=0.1):
        """Generate potential outcomes using a realistic treatment effect model."""
        n_samples = len(x)
        treatment_effect = np.zeros(n_samples)
        
        # Use numeric features for treatment effect (first 5 features)
        for i in range(5):
            # Main effect
            treatment_effect += np.random.normal(0, noise_std) * x[:, i]
            
            # Interaction effects
            for j in range(i+1, 5):
                interaction = x[:, i] * x[:, j]
                treatment_effect += np.random.normal(0, noise_std/2) * interaction
        
        # Scale treatment effect to reasonable range
        treatment_effect = 0.1 * treatment_effect
        
        # Generate potential outcomes
        potential_y = np.zeros((n_samples, 2))
        
        # Control outcome (y0)
        potential_y[:, 0] = np.where(
            t == 0,
            y,  # For control group, use observed outcome
            y - treatment_effect  # For treated group, subtract treatment effect
        )
        
        # Treatment outcome (y1)
        potential_y[:, 1] = np.where(
            t == 1,
            y,  # For treated group, use observed outcome
            y + treatment_effect  # For control group, add treatment effect
        )
        
        # Ensure outcomes are in [0,1] range
        potential_y = np.clip(potential_y, 0, 1)
        
        return potential_y
    
    def rescale_outcome(self, y):
        """Rescale normalized outcomes back to original scale with validation."""
        if self.y_min is None or self.y_max is None:
            raise ValueError("Scaling parameters not initialized")
        
        # Input validation
        if isinstance(y, list):
            y = np.array(y)
            
        # Check for values outside expected range
        tolerance = 1e-6
        if np.any(y < -tolerance) or np.any(y > 1 + tolerance):
            print(f"Warning: Values outside expected range [0,1]: min={np.min(y)}, max={np.max(y)}")
        
        # Clip values to ensure they're in [0,1] range
        y = np.clip(y, 0, 1)
        
        return y * (self.y_max - self.y_min) + self.y_min