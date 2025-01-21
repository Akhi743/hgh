"""GANITE Codebase for LaLonde Dataset.

Last updated Date: January 17th 2025
Code adapted for LaLonde dataset

-----------------------------

data_loading.py

Note: Load real-world individualized treatment effects estimation datasets

(1) data_loading_lalonde: Load lalonde data.
"""

# Necessary packages
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

def data_loading_lalonde(train_rate = 0.8):
    """Load LaLonde data.
    
    Args:
        - train_rate: the ratio of training data
        
    Returns:
        - train_x: features in training data
        - train_t: treatments in training data
        - train_y: observed outcomes in training data
        - train_potential_y: potential outcomes in training data
        - test_x: features in testing data
        - test_potential_y: potential outcomes in testing data      
    """
    
    # Load original data
    try:
        ori_data = pd.read_csv("Dataset/lalonde.csv")
        print("Available columns:", ori_data.columns.tolist())
    except:
        raise ValueError("Cannot find Dataset/lalonde.csv")
    
    # Create feature matrix
    feature_columns = ['age', 'educ', 'race', 'married', 'nodegree', 're74', 're75']
    data = ori_data[feature_columns].copy()
    
    # Create dummy variables for categorical variables
    data = pd.get_dummies(data, columns=['race'], prefix=['race'])
    
    # Define continuous and binary features
    continuous_features = ['age', 'educ', 're74', 're75']
    binary_features = ['married', 'nodegree'] + [col for col in data.columns if col.startswith('race_')]
    
    # Apply StandardScaler to continuous features
    scaler = StandardScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    
    # Convert to numpy array
    x = data.values
    no, dim = x.shape
    
    # Get treatment and outcome
    t = ori_data['treat'].values
    y = ori_data['re78'].values
    
    # Create potential outcomes
    potential_y = np.zeros((no, 2))
    
    # Observed factuals
    treated_idx = t == 1
    control_idx = t == 0
    
    potential_y[treated_idx, 1] = y[treated_idx]  # Observed treated outcomes
    potential_y[control_idx, 0] = y[control_idx]  # Observed control outcomes
    
    # Estimate counterfactuals
    treated_mean = np.mean(y[treated_idx])
    control_mean = np.mean(y[control_idx])
    
    # For control units, estimate treated outcome
    effect_ratio = treated_mean / control_mean if control_mean != 0 else 1
    potential_y[control_idx, 1] = y[control_idx] * effect_ratio
    
    # For treated units, estimate control outcome
    inverse_ratio = control_mean / treated_mean if treated_mean != 0 else 1
    potential_y[treated_idx, 0] = y[treated_idx] * inverse_ratio
    
    # Ensure non-negative outcomes
    potential_y = np.maximum(potential_y, 0)
    
    # Reshape treatment and outcome
    t = t.reshape(-1)
    y = y.reshape(-1)
    
    # Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_rate * no)]
    test_idx = idx[int(train_rate * no):]
    
    train_x = x[train_idx, :]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx, :]
    
    test_x = x[test_idx, :]
    test_potential_y = potential_y[test_idx, :]
    
    # Print shapes for debugging
    print(f"Features shape: {x.shape}")
    print(f"Feature columns: {list(data.columns)}")
    
    return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y
