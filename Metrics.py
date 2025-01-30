import numpy as np
from Utils import Utils

class Metrics:
    @staticmethod
    def PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        """
        Calculate Population Expected Heterogeneous Effect (PEHE) for social risk scores
        """
        PEHE_val = np.sqrt(np.mean(np.square(
            (Utils.convert_to_col_vector(y1_true_np) - Utils.convert_to_col_vector(y0_true_np)) - 
            (Utils.convert_to_col_vector(y1_hat_np) - Utils.convert_to_col_vector(y0_hat_np))
        )))
        return PEHE_val

    @staticmethod
    def ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        """
        Calculate Average Treatment Effect (ATE) for social risk scores
        """
        ATE_val = np.abs(np.mean(y1_true_np - y0_true_np) - np.mean(y1_hat_np - y0_hat_np))
        return ATE_val
        
    @staticmethod
    def RMSE(y_true, y_pred):
        """
        Calculate Root Mean Square Error for social risk predictions
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
        
    @staticmethod
    def MAE(y_true, y_pred):
        """
        Calculate Mean Absolute Error for social risk predictions
        """
        return np.mean(np.abs(y_true - y_pred))