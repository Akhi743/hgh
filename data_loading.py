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
        data = pd.read_csv("Dataset/lalonde.csv")
    except Exception as e:
        raise ValueError(f"Error loading Dataset/lalonde.csv: {str(e)}")
    
    # Drop the unnamed index column
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # Create dummy variables for race
    race_dummies = pd.get_dummies(data['race'], prefix='race')
    
    # Drop original race column and join with dummy variables
    data = data.drop('race', axis=1)
    data = pd.concat([data, race_dummies], axis=1)
    
    # Define features (excluding treatment and outcome)
    feature_columns = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + list(race_dummies.columns)
    
    # Extract features
    x = data[feature_columns].values
    no, dim = x.shape
    
    # Normalize continuous features (age, educ, re74, re75)
    continuous_features = [0, 1, 4, 5]  # indices of continuous features
    for i in continuous_features:
        mean_val = np.mean(x[:, i])
        std_val = np.std(x[:, i])
        x[:, i] = (x[:, i] - mean_val) / (std_val + 1e-8)
    
    # Get treatment and outcome
    t = data['treat'].values
    y = data['re78'].values
    
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
    
    # Print shape information for debugging
    print(f"Features shape: {x.shape}")
    print(f"Feature columns: {feature_columns}")
    print(f"Created {len(race_dummies.columns)} race dummy variables")
    
    return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y