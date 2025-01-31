import numpy as np
from scipy import stats

class Metrics:
    def __init__(self, y_min, y_max):
        """Initialize metrics calculator with scaling parameters."""
        if y_min >= y_max:
            raise ValueError("y_min must be less than y_max")
        self.y_min = y_min
        self.y_max = y_max
    
    def rescale_outcome(self, y):
        """Rescale normalized outcomes back to original scale with validation."""
        if self.y_min is None or self.y_max is None:
            raise ValueError("Scaling parameters not initialized")
        
        # Input validation
        if isinstance(y, list):
            y = np.array(y)
        
        # Print pre-rescaling statistics
        print(f"\nPre-rescaling statistics:")
        print(f"Input range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        print(f"Mean: {np.mean(y):.4f}")
        print(f"Std: {np.std(y):.4f}")
        
        # Check for values outside expected range
        tolerance = 1e-6
        if np.any(y < -tolerance) or np.any(y > 1 + tolerance):
            outside_range = np.sum((y < -tolerance) | (y > 1 + tolerance))
            print(f"Warning: {outside_range} values outside expected range [0,1]")
            print(f"Values < 0: {np.sum(y < -tolerance)}")
            print(f"Values > 1: {np.sum(y > 1 + tolerance)}")
        
        # Clip values to ensure they're in [0,1] range
        y_clipped = np.clip(y, 0, 1)
        
        # Rescale
        y_rescaled = y_clipped * (self.y_max - self.y_min) + self.y_min
        
        # Print post-rescaling statistics
        print(f"\nPost-rescaling statistics:")
        print(f"Output range: [{np.min(y_rescaled):.4f}, {np.max(y_rescaled):.4f}]")
        print(f"Mean: {np.mean(y_rescaled):.4f}")
        print(f"Std: {np.std(y_rescaled):.4f}")
        
        return y_rescaled
    
    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        """Calculate mean and confidence interval."""
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, h
    
    def calculate_metrics(self, y_true, y_pred, verbose=True):
        """Calculate metrics with enhanced diagnostics."""
        print("\n=== Starting Metrics Calculation ===")
        print(f"Original scale range: [{self.y_min:.4f}, {self.y_max:.4f}]")
        
        print("\nAnalyzing true outcomes (y_true):")
        y_true_rescaled = self.rescale_outcome(y_true)
        
        print("\nAnalyzing predicted outcomes (y_pred):")
        y_pred_rescaled = self.rescale_outcome(y_pred)
        
        # Calculate and analyze treatment effects
        true_effect = y_true_rescaled[:, 1] - y_true_rescaled[:, 0]
        pred_effect = y_pred_rescaled[:, 1] - y_pred_rescaled[:, 0]
        
        print("\nTreatment Effects Analysis:")
        print(f"True effect - Range: [{np.min(true_effect):.4f}, {np.max(true_effect):.4f}]")
        print(f"True effect - Mean: {np.mean(true_effect):.4f}")
        print(f"True effect - Std: {np.std(true_effect):.4f}")
        print(f"Pred effect - Range: [{np.min(pred_effect):.4f}, {np.max(pred_effect):.4f}]")
        print(f"Pred effect - Mean: {np.mean(pred_effect):.4f}")
        print(f"Pred effect - Std: {np.std(pred_effect):.4f}")
        
        # Calculate metrics
        pehe_data = np.square(true_effect - pred_effect)
        pehe_value = np.sqrt(np.mean(pehe_data))
        _, pehe_ci = self.mean_confidence_interval(pehe_data)
        
        ate_value = np.abs(np.mean(true_effect) - np.mean(pred_effect))
        ate_data = np.abs(true_effect - pred_effect)
        _, ate_ci = self.mean_confidence_interval(ate_data)
        
        mse = np.mean(np.square(y_true_rescaled - y_pred_rescaled))
        mae = np.mean(np.abs(y_true_rescaled - y_pred_rescaled))
        rmse = np.sqrt(mse)
        
        # Calculate R-squared for additional context
        ss_tot = np.sum(np.square(y_true_rescaled - np.mean(y_true_rescaled)))
        ss_res = np.sum(np.square(y_true_rescaled - y_pred_rescaled))
        r2 = 1 - (ss_res / ss_tot)
        
        print("\nError Analysis:")
        print(f"Error range: [{np.min(y_true_rescaled - y_pred_rescaled):.4f}, "
              f"{np.max(y_true_rescaled - y_pred_rescaled):.4f}]")
        print(f"Error mean: {np.mean(y_true_rescaled - y_pred_rescaled):.4f}")
        print(f"Error std: {np.std(y_true_rescaled - y_pred_rescaled):.4f}")
        print(f"R-squared: {r2:.4f}")
        
        metrics_dict = {
            'PEHE': {
                'value': pehe_value,
                'confidence_interval': pehe_ci
            },
            'ATE': {
                'value': ate_value,
                'confidence_interval': ate_ci
            },
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics_dict