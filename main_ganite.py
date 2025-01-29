"""GANITE Codebase.

main_ganite.py

(1) Import data
(2) Train GANITE & Estimate potential outcomes
(3) Evaluate the performances
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. GANITE model
from ganite import ganite
# 2. Data loading
from data_loading import data_loading_lalonde
# 3. Metrics
from metrics import PEHE, ATE

def run_experiment(args, random_seed):
    """Run a single experiment with given random seed."""
    ## Data loading
    train_x, train_t, train_y, train_potential_y, test_x, test_potential_y, y_scaler = \
    data_loading_lalonde(args.train_rate, random_seed)
    
    ## Potential outcome estimations by GANITE
    # Set network parameters
    parameters = dict()
    parameters['h_dim'] = args.h_dim
    parameters['iteration'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['alpha'] = args.alpha
    
    test_y_hat = ganite(train_x, train_t, train_y, test_x, parameters)
    
    # Calculate metrics
    test_PEHE = PEHE(test_potential_y, test_y_hat, y_scaler)
    test_ATE = ATE(test_potential_y, test_y_hat, y_scaler)
    
    # Calculate training metrics
    train_y_hat = ganite(train_x, train_t, train_y, train_x, parameters)
    train_PEHE = PEHE(train_potential_y, train_y_hat, y_scaler)
    train_ATE = ATE(train_potential_y, train_y_hat, y_scaler)
    
    return test_PEHE, test_ATE, train_PEHE, train_ATE

def main(args):
    """Main function for GANITE experiments.
    
    Args:
        - data_name: lalonde
        - train_rate: ratio of training data
        - Network parameters (should be optimized for different datasets)
            - h_dim: hidden dimensions
            - iteration: number of training iterations
            - batch_size: the number of samples in each batch
            - alpha: hyper-parameter to adjust the loss importance
    """
    # Run multiple experiments
    n_experiments = 10
    test_PEHEs = []
    test_ATEs = []
    train_PEHEs = []
    train_ATEs = []
    
    for i in range(n_experiments):
        print(f"\nRunning experiment {i+1}/{n_experiments}")
        test_PEHE, test_ATE, train_PEHE, train_ATE = run_experiment(args, random_seed=i)
        test_PEHEs.append(test_PEHE)
        test_ATEs.append(test_ATE)
        train_PEHEs.append(train_PEHE)
        train_ATEs.append(train_ATE)
    
    # Calculate means and standard deviations
    test_PEHE_mean = np.mean(test_PEHEs)
    test_PEHE_std = np.std(test_PEHEs)
    test_ATE_mean = np.mean(test_ATEs)
    test_ATE_std = np.std(test_ATEs)
    
    train_PEHE_mean = np.mean(train_PEHEs)
    train_PEHE_std = np.std(train_PEHEs)
    train_ATE_mean = np.mean(train_ATEs)
    train_ATE_std = np.std(train_ATEs)
    
    # Print results
    print("\nFinal Results:")
    print("-" * 40)
    print(f"GANITE, PEHE_out: ${test_PEHE_mean:.2f}, SD: ${test_PEHE_std:.2f}")
    print(f"GANITE, ATE Metric_out: ${test_ATE_mean:.2f}, SD: ${test_ATE_std:.2f}")
    print("-" * 40)
    print(f"GANITE, PEHE_in: ${train_PEHE_mean:.2f}, SD: ${train_PEHE_std:.2f}")
    print(f"GANITE, ATE Metric_in: ${train_ATE_mean:.2f}, SD: ${train_ATE_std:.2f}")

if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['lalonde'],
        default='lalonde',
        type=str)
    parser.add_argument(
        '--train_rate',
        help='the ratio of training data',
        default=0.8,
        type=float)
    parser.add_argument(
        '--h_dim',
        help='hidden state dimensions (should be optimized)',
        default=100,  # Increased from 30
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=20000,  # Increased from 10000
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,  # Changed from 256
        type=int)
    parser.add_argument(
        '--alpha',
        help='hyper-parameter to adjust the loss importance (should be optimized)',
        default=2,  # Changed from 1
        type=int)
    
    args = parser.parse_args()
    
    # Calls main function
    main(args)