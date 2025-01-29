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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def data_loading_lalonde(train_rate=0.8, random_seed=None):
    """Load LaLonde data.
    
    Args:
        - train_rate: the ratio of training data
        - random_seed: random seed for reproducibility
        
    Returns:
        - train_x: features in training data
        - train_t: treatments in training data
        - train_y: observed outcomes in training data
        - train_potential_y: potential outcomes in training data
        - test_x: features in testing data
        - test_potential_y: potential outcomes in testing data
        - scalers: dictionary containing fitted scalers for inverse transform
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Load original data
    data = pd.read_csv("lalonde.csv")
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # Create dummy variables for race (one-hot encoding)
    race_dummies = pd.get_dummies(data['race'], prefix='race')
    
    # Drop original race column and join with dummy variables
    data = data.drop('race', axis=1)
    data = pd.concat([data, race_dummies], axis=1)
    
    # Define features (excluding treatment and outcome)
    feature_columns = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + list(race_dummies.columns)
    
    # Extract features
    x = data[feature_columns].values
    no, dim = x.shape
    
    # Initialize scalers
    scalers = {
        'continuous': StandardScaler(),  # For continuous features
        'earnings': MinMaxScaler(),      # For earnings (re74, re75)
        'outcome': MinMaxScaler()        # For the outcome (re78)
    }
    
    # Normalize continuous features
    continuous_features = ['age', 'educ']
    cont_indices = [feature_columns.index(feat) for feat in continuous_features]
    x[:, cont_indices] = scalers['continuous'].fit_transform(x[:, cont_indices])
    
    # Normalize earnings features (re74, re75)
    earnings_features = ['re74', 're75']
    earnings_indices = [feature_columns.index(feat) for feat in earnings_features]
    x[:, earnings_indices] = scalers['earnings'].fit_transform(x[:, earnings_indices])
    
    # Get treatment and outcome
    t = data['treat'].values
    y = data['re78'].values
    
    # Store original y for potential outcomes calculation
    y_orig = y.copy()
    
    # Scale outcome (re78) to [0,1] range
    y = scalers['outcome'].fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create potential outcomes
    potential_y = np.zeros((no, 2))
    
    # Observed factuals
    treated_idx = t == 1
    control_idx = t == 0
    
    # Calculate mean outcomes for each group (in original scale)
    treated_mean = np.mean(y_orig[treated_idx])
    control_mean = np.mean(y_orig[control_idx])
    
    # Store observed outcomes (in scaled form)
    potential_y[treated_idx, 1] = y[treated_idx]  # Observed treated outcomes
    potential_y[control_idx, 0] = y[control_idx]  # Observed control outcomes
    
    # Estimate counterfactuals using matching on covariates
    from sklearn.neighbors import NearestNeighbors
    
    # For control units (estimate treated outcomes)
    control_x = x[control_idx]
    treated_x = x[treated_idx]
    treated_y = y[treated_idx]
    
    if len(treated_x) > 0 and len(control_x) > 0:
        nbrs_treated = NearestNeighbors(n_neighbors=5).fit(treated_x)
        distances, indices = nbrs_treated.kneighbors(control_x)
        # Use exponential weights
        weights = np.exp(-distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        potential_y[control_idx, 1] = np.sum(treated_y[indices] * weights, axis=1)
    
    # For treated units (estimate control outcomes)
    control_y = y[control_idx]
    
    if len(control_x) > 0 and len(treated_x) > 0:
        nbrs_control = NearestNeighbors(n_neighbors=5).fit(control_x)
        distances, indices = nbrs_control.kneighbors(treated_x)
        # Use exponential weights
        weights = np.exp(-distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        potential_y[treated_idx, 0] = np.sum(control_y[indices] * weights, axis=1)
    
    # Train/test division using stratification
    train_idx, test_idx = train_test_split(
        np.arange(no),
        train_size=train_rate,
        random_state=random_seed,
        stratify=t  # Stratify by treatment to maintain treatment ratio
    )
    
    train_x = x[train_idx, :]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx, :]
    
    test_x = x[test_idx, :]
    test_potential_y = potential_y[test_idx, :]
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {no}")
    print(f"Treatment group: {np.sum(t == 1)} samples ({np.mean(t):.3f})")
    print(f"Control group: {np.sum(t == 0)} samples ({1-np.mean(t):.3f})")
    print(f"\nOriginal Earnings Statistics:")
    print(f"Treated mean: ${treated_mean:.2f}")
    print(f"Control mean: ${control_mean:.2f}")
    print(f"Raw ATE: ${treated_mean - control_mean:.2f}")
    print(f"\nTraining samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    print(f"Training treatment ratio: {train_t.mean():.3f}")
    
    return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y, scalers
