"""GANITE Codebase.

metrics.py

Note: Metric functions for GANITE.
Reference: Jennifer L Hill, "Bayesian nonparametric modeling for causal inference", 
Journal of Computational and Graphical Statistics, 2011.

(1) PEHE: Precision in Estimation of Heterogeneous Effect
(2) ATE: Average Treatment Effect
"""

# Necessary packages
import numpy as np

def normalize_outcomes(y):
    """Normalize outcomes to [0,1] range.
    """
    y_min = np.min(y)
    y_max = np.max(y)
    return (y - y_min) / (y_max - y_min + 1e-8)

def PEHE(y, y_hat):
    """Compute normalized Precision in Estimation of Heterogeneous Effect.
    
    Args:
        - y: potential outcomes
        - y_hat: estimated potential outcomes
        
    Returns:
        - PEHE_val: computed PEHE
    """
    # Normalize true outcomes
    y_norm = normalize_outcomes(y)
    # Normalize predicted outcomes
    y_hat_norm = normalize_outcomes(y_hat)
    
    # Calculate PEHE on normalized values
    PEHE_val = np.mean(np.square((y_norm[:,1] - y_norm[:,0]) - (y_hat_norm[:,1] - y_hat_norm[:,0])))
    return np.sqrt(PEHE_val)

def ATE(y, y_hat):
    """Compute normalized Average Treatment Effect.
    
    Args:
        - y: potential outcomes
        - y_hat: estimated potential outcomes
        
    Returns:
        - ATE_val: computed ATE
    """
    # Normalize true outcomes
    y_norm = normalize_outcomes(y)
    # Normalize predicted outcomes
    y_hat_norm = normalize_outcomes(y_hat)
    
    # Calculate ATE on normalized values
    ATE_val = np.abs(np.mean(y_norm[:,1] - y_norm[:,0]) - np.mean(y_hat_norm[:,1] - y_hat_norm[:,0]))
    return ATE_val