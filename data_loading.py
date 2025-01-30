import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def data_loading_custom(train_data_split_rate=0.8):
    """Load custom dataset for GANITE with proper preprocessing including one-hot encoding.
    
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
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop=None, sparse_output=False))  # Changed to keep all categories
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the feature preprocessing
    x = preprocessor.fit_transform(data[numeric_features + categorical_features])
    
    # Get feature names after transformation
    numeric_feature_names = numeric_features
    categorical_feature_names = []
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
        categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_feature_names + list(categorical_feature_names)
    
    no, dim = x.shape
    print(f"Number of samples: {no}")
    print(f"Number of features after one-hot encoding: {dim}")
    print(f"Feature names: {all_feature_names}")
    
    # Treatment assignment
    t = data['grp_binary'].values
    
    # Outcome
    y = data['social_risk_score'].values
    
    # Normalize outcome
    y = (y - y.min()) / (y.max() - y.min())
    
    # Generate potential outcomes with feature interactions
    potential_y = np.zeros((no, 2))
    
    # Base effect using all features
    base_effect = np.dot(x, np.random.normal(0, 0.1, (dim, 1))).flatten()
    
    # Treatment effect incorporating feature interactions
    treatment_effect = np.dot(x, np.random.normal(0, 0.1, (dim, 1))).flatten()
    
    # Add interaction terms between numeric features
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            interaction = x[:, i] * x[:, j]
            treatment_effect += np.random.normal(0, 0.05) * interaction
    
    # Generate potential outcomes
    potential_y[:, 0] = y * (1 - t) + (y - treatment_effect) * t  # Outcome if not treated
    potential_y[:, 1] = y * t + (y + treatment_effect) * (1 - t)  # Outcome if treated
    
    # Ensure outcomes are bounded between 0 and 1
    potential_y = np.clip(potential_y, 0, 1)
    
    # Train/test split
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_data_split_rate * no)]
    test_idx = idx[int(train_data_split_rate * no):]
    
    train_x = x[train_idx, :]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx, :]
    
    test_x = x[test_idx, :]
    test_potential_y = potential_y[test_idx, :]
    
    return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y