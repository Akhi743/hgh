import numpy as np
from scipy import stats

def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h

def PEHE(y, y_hat):
    """Precision in Estimation of Heterogeneous Effect.
    
    Args:
        y: true potential outcomes
        y_hat: estimated potential outcomes
    
    Returns:
        PEHE value and confidence interval
    """
    data = np.square((y[:, 1] - y[:, 0]) - (y_hat[:, 1] - y_hat[:, 0]))
    PEHE_val = np.sqrt(np.mean(data))
    interval = mean_confidence_interval(data)
    return PEHE_val, interval

def ATE(y, y_hat):
    """Average Treatment Effect.
    
    Args:
        y: true potential outcomes
        y_hat: estimated potential outcomes
    
    Returns:
        ATE value and confidence interval
    """
    true_effect = y[:, 1] - y[:, 0]
    estimated_effect = y_hat[:, 1] - y_hat[:, 0]
    
    data = np.abs(true_effect - estimated_effect)
    ate = np.abs(np.mean(true_effect) - np.mean(estimated_effect))
    
    interval = mean_confidence_interval(data)
    return ate, interval