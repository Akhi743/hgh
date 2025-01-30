"""GANITE Implementation for Custom Dataset
"""

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ganite_torch import ganite_torch
from data_loading import data_loading_custom
from metrics import PEHE, ATE
import os
from datetime import datetime

def create_result_dir(name, parameters):
    """Create directory for results"""
    if not os.path.exists(f"results/{name}"):
        os.makedirs(f"results/{name}")
        os.makedirs(f"results/{name}/logs")
        os.makedirs(f"results/{name}/models")

    with open(f"results/{name}/parameters.txt", "w") as f:
        f.write(f"Experiment run at: {datetime.now()}\n\n")
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

def main(args):
    """Main function for GANITE experiments.
    """
    # Data loading
    train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
        data_loading_custom(args.train_rate)

    print('Dataset is ready.')
    print(f'Training samples: {train_x.shape[0]}')
    print(f'Testing samples: {test_x.shape[0]}')
    print(f'Number of features: {train_x.shape[1]}')

    # Set network parameters
    parameters = {
        'h_dim': args.h_dim,
        'iteration': args.iteration,
        'batch_size': args.batch_size,
        'alpha': args.alpha,
        'beta': args.beta,
        'lr': args.lr
    }
    
    flags = {
        'dropout': args.dropout,
        'adamw': args.adamw
    }

    # Create results directory
    create_result_dir(args.name, parameters)

    # Run GANITE
    print('\nStarting GANITE training...')
    test_y_hat = ganite_torch(train_x, train_t, train_y, test_x, 
                             train_potential_y, test_potential_y, 
                             parameters, args.name, flags)
    print('Finished GANITE training and potential outcome estimations')

    # Calculate metrics
    metric_results = {}

    # PEHE
    test_PEHE, interval = PEHE(test_potential_y, test_y_hat)
    metric_results['PEHE'] = test_PEHE
    metric_results['PEHE_interval'] = interval

    # ATE
    test_ATE, interval = ATE(test_potential_y, test_y_hat)
    metric_results['ATE'] = test_ATE
    metric_results['ATE_interval'] = interval

    # Print and save results
    print('\nResults:')
    print(f'PEHE: {test_PEHE:.4f} ± {interval[1]:.4f}')
    print(f'ATE: {test_ATE:.4f} ± {interval[1]:.4f}')

    with open(f"results/{args.name}/results.txt", "w") as f:
        f.write(str(metric_results))

    return test_y_hat, metric_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_rate', default=0.8, type=float,
                       help='ratio of training data')
    parser.add_argument('--h_dim', default=30, type=int,
                       help='hidden state dimensions')
    parser.add_argument('--iteration', default=10000, type=int,
                       help='number of training iterations')
    parser.add_argument('--batch_size', default=128, type=int,
                       help='batch size')
    parser.add_argument('--alpha', default=1.0, type=float,
                       help='generator loss hyperparameter')
    parser.add_argument('--beta', default=1.0, type=float,
                       help='inference loss hyperparameter')
    parser.add_argument('--name', default='experiment1', type=str,
                       help='experiment name')
    parser.add_argument('--lr', default=1e-3, type=float,
                       help='learning rate')
    parser.add_argument('--dropout', action='store_true',
                       help='use dropout')
    parser.add_argument('--adamw', action='store_true',
                       help='use AdamW optimizer')

    args = parser.parse_args()
    
    # Run main function
    test_y_hat, metrics = main(args)