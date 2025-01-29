import numpy as np
from Utils import Utils

class Metrics:
    @staticmethod
    def PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        """
        Calculate the Precision in Estimation of Heterogeneous Effect (PEHE)
        τ(x) = E[Y|T=1, X=x] - E[Y|T=0, X=x]
        PEHE = sqrt(1/n * sum((τ_true - τ_pred)^2))
        
        Args:
            y1_true_np: True outcomes under treatment
            y0_true_np: True outcomes under control
            y1_hat_np: Predicted outcomes under treatment
            y0_hat_np: Predicted outcomes under control
            
        Returns:
            float: PEHE value
        """
        # Convert to column vectors
        y1_true = Utils.convert_to_col_vector(y1_true_np)
        y0_true = Utils.convert_to_col_vector(y0_true_np)
        y1_hat = Utils.convert_to_col_vector(y1_hat_np)
        y0_hat = Utils.convert_to_col_vector(y0_hat_np)
        
        # Calculate true and predicted ITEs
        ite_true = y1_true - y0_true
        ite_pred = y1_hat - y0_hat
        
        # Calculate PEHE
        pehe = np.sqrt(np.mean(np.square(ite_true - ite_pred)))
        return pehe

    @staticmethod
    def ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        """
        Calculate the Average Treatment Effect error
        ATE = |E[τ_true] - E[τ_pred]|
        where τ = Y1 - Y0
        
        Args:
            y1_true_np: True outcomes under treatment
            y0_true_np: True outcomes under control
            y1_hat_np: Predicted outcomes under treatment
            y0_hat_np: Predicted outcomes under control
            
        Returns:
            float: ATE error value
        """
        # Convert to column vectors
        y1_true = Utils.convert_to_col_vector(y1_true_np)
        y0_true = Utils.convert_to_col_vector(y0_true_np)
        y1_hat = Utils.convert_to_col_vector(y1_hat_np)
        y0_hat = Utils.convert_to_col_vector(y0_hat_np)
        
        # Calculate true and predicted ATEs
        ate_true = np.mean(y1_true - y0_true)
        ate_pred = np.mean(y1_hat - y0_hat)
        
        # Calculate absolute ATE error
        ate_error = np.abs(ate_true - ate_pred)
        return ate_error

    @staticmethod
    def bias(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        """
        Calculate bias in treatment effect estimation
        bias = E[τ_pred - τ_true]
        
        Args:
            y1_true_np: True outcomes under treatment
            y0_true_np: True outcomes under control
            y1_hat_np: Predicted outcomes under treatment
            y0_hat_np: Predicted outcomes under control
            
        Returns:
            float: Bias value
        """
        # Convert to column vectors
        y1_true = Utils.convert_to_col_vector(y1_true_np)
        y0_true = Utils.convert_to_col_vector(y0_true_np)
        y1_hat = Utils.convert_to_col_vector(y1_hat_np)
        y0_hat = Utils.convert_to_col_vector(y0_hat_np)
        
        # Calculate ITEs
        ite_true = y1_true - y0_true
        ite_pred = y1_hat - y0_hat
        
        # Calculate bias
        bias_val = np.mean(ite_pred - ite_true)
        return bias_val

    @staticmethod
    def evaluate_predictions(y1_true, y0_true, y1_pred, y0_pred, scaler=None):
        """
        Evaluate predictions using multiple metrics with option to convert back to original scale
        
        Args:
            y1_true: True outcomes under treatment
            y0_true: True outcomes under control
            y1_pred: Predicted outcomes under treatment
            y0_pred: Predicted outcomes under control
            scaler: Scaler object for converting back to original scale (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # If scaler is provided, convert predictions to original scale
        if scaler is not None:
            y1_true = scaler.inverse_transform(y1_true.reshape(-1, 1))
            y0_true = scaler.inverse_transform(y0_true.reshape(-1, 1))
            y1_pred = scaler.inverse_transform(y1_pred.reshape(-1, 1))
            y0_pred = scaler.inverse_transform(y0_pred.reshape(-1, 1))

        # Calculate all metrics
        pehe = Metrics.PEHE(y1_true, y0_true, y1_pred, y0_pred)
        ate_error = Metrics.ATE(y1_true, y0_true, y1_pred, y0_pred)
        bias_val = Metrics.bias(y1_true, y0_true, y1_pred, y0_pred)

        # Calculate additional statistics
        ite_pred = y1_pred - y0_pred
        ate_pred = np.mean(ite_pred)
        ite_std = np.std(ite_pred)

        return {
            'PEHE': pehe,
            'ATE_error': ate_error,
            'Bias': bias_val,
            'ATE_pred': ate_pred,
            'ITE_std': ite_std
        }

    @staticmethod
    def print_metrics(metrics_dict, dollar_values=True):
        """
        Print metrics in a formatted way
        
        Args:
            metrics_dict: Dictionary containing metrics
            dollar_values: Whether to format values as dollars (optional)
        """
        print("\nEvaluation Metrics:")
        print("-" * 40)
        
        if dollar_values:
            print(f"PEHE: ${metrics_dict['PEHE']:.2f}")
            print(f"ATE Error: ${metrics_dict['ATE_error']:.2f}")
            print(f"Bias: ${metrics_dict['Bias']:.2f}")
            print(f"Predicted ATE: ${metrics_dict['ATE_pred']:.2f}")
            print(f"ITE Standard Deviation: ${metrics_dict['ITE_std']:.2f}")
        else:
            print(f"PEHE: {metrics_dict['PEHE']:.4f}")
            print(f"ATE Error: {metrics_dict['ATE_error']:.4f}")
            print(f"Bias: {metrics_dict['Bias']:.4f}")
            print(f"Predicted ATE: {metrics_dict['ATE_pred']:.4f}")
            print(f"ITE Standard Deviation: {metrics_dict['ITE_std']:.4f}")
            
        print("-" * 40)
