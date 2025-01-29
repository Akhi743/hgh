"""Metrics for evaluation.

(1) PEHE: Precision in Estimation of Heterogeneous Effect 
(2) ATE: Average Treatment Effect
"""

import numpy as np

def PEHE(y, y_hat, scalers=None):
    """Compute Precision in Estimation of Heterogeneous Effect.
    
    Args:
        - y: potential outcomes
        - y_hat: estimated potential outcomes
        - scalers: dictionary containing fitted scalers
        
    Returns:
        - PEHE_val: computed PEHE in original scale (dollars for LaLonde)
    """
    # If scalers provided, transform back to original scale
    if scalers is not None:
        y = scalers['outcome'].inverse_transform(y)
        y_hat = scalers['outcome'].inverse_transform(y_hat)
    
    # Calculate PEHE
    PEHE_val = np.mean(np.square((y[:,1] - y[:,0]) - (y_hat[:,1] - y_hat[:,0])))
    return np.sqrt(PEHE_val)

def ATE(y, y_hat, scalers=None):
    """Compute Average Treatment Effect.
    
    Args:
        - y: potential outcomes
        - y_hat: estimated potential outcomes
        - scalers: dictionary containing fitted scalers
        
    Returns:
        - ATE_val: computed ATE in original scale (dollars for LaLonde)
    """
    # If scalers provided, transform back to original scale
    if scalers is not None:
        y = scalers['outcome'].inverse_transform(y)
        y_hat = scalers['outcome'].inverse_transform(y_hat)
    
    # Calculate true and estimated effects
    true_effect = np.mean(y[:,1] - y[:,0])
    estimated_effect = np.mean(y_hat[:,1] - y_hat[:,0])
    
    # Calculate absolute difference in ATEs
    ATE_val = np.abs(true_effect - estimated_effect)
    
    # Print detailed information
    print(f"\nDetailed ATE Analysis:")
    print(f"True Effect: ${true_effect:.2f}")
    print(f"Estimated Effect: ${estimated_effect:.2f}")
    print(f"ATE Difference: ${ATE_val:.2f}")
    
    return ATE_val