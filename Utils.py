import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn
import torch

class NormalNLLLoss:
    """
    Calculate the negative log likelihood of normal distribution.
    This needs to be minimized.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, noise_z, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (noise_z - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())
        return nll

class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def convert_to_col_vector(np_arr):
        return np_arr.reshape(np_arr.shape[0], 1)

    @staticmethod
    def test_train_split(np_train_X, np_train_T, np_train_yf, np_train_ycf, split_size=0.8):
        return sklearn.train_test_split(np_train_X, np_train_T, np_train_yf, np_train_ycf,
                                      train_size=split_size)

    @staticmethod
    def convert_to_tensor(X, T, Y_f, Y_cf):
        # Ensure all inputs are float32 type
        X = X.astype(np.float32)
        T = T.astype(np.float32)
        Y_f = Y_f.astype(np.float32)
        Y_cf = Y_cf.astype(np.float32)
        
        # Convert to tensors
        tensor_x = torch.from_numpy(X)
        tensor_T = torch.from_numpy(T)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)

        processed_dataset = torch.utils.data.TensorDataset(tensor_x,
                                                         tensor_T,
                                                         tensor_y_f,
                                                         tensor_y_cf)
        return processed_dataset

    @staticmethod
    def concat_np_arr(X, Y, axis=1):
        return np.concatenate((X, Y), axis)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def write_to_csv(file_name, list_to_write):
        pd.DataFrame.from_dict(
            list_to_write,
            orient='columns'
        ).to_csv(file_name)

    @staticmethod
    def vae_loss(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())