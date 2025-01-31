import argparse
import os
from datetime import datetime
from data_loading import DataLoader
from metrics import Metrics
from ganite_torch import ganite_torch

def create_result_dir(name, parameters):
    """Create directory for results and save parameters."""
    if not os.path.exists(f"results/{name}"):
        os.makedirs(f"results/{name}")
        os.makedirs(f"results/{name}/logs")
        os.makedirs(f"results/{name}/models")
    
    with open(f"results/{name}/parameters.txt", "w") as f:
        f.write(f"Experiment run at: {datetime.now()}\n\n")
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

def main(args):
    """Main function for GANITE experiments."""
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load and preprocess data
    train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
        data_loader.data_loading_custom(args.train_rate)
    
    print('Dataset is ready.')
    print(f'Training samples: {train_x.shape[0]}')
    print(f'Testing samples: {test_x.shape[0]}')
    print(f'Number of features: {train_x.shape[1]}')
    print(f'Original scale - Min: {data_loader.y_min:.4f}, Max: {data_loader.y_max:.4f}')
    
    # Set network parameters
    parameters = {
        'h_dim': args.h_dim,
        'iteration': args.iteration,
        'batch_size': args.batch_size,
        'alpha': args.alpha,
        'beta': args.beta,
        'lr': args.lr,
        'y_min': data_loader.y_min,
        'y_max': data_loader.y_max
    }
    
    flags = {
        'dropout': args.dropout,
        'adamw': args.adamw
    }
    
    # Create results directory
    create_result_dir(args.name, parameters)
    
    # Train GANITE
    print('\nStarting GANITE training...')
    test_y_hat = ganite_torch(
        train_x, train_t, train_y, test_x, train_potential_y, test_potential_y,
        parameters, args.name, flags
    )
    print('Finished GANITE training and potential outcome estimations')
    
    # Initialize metrics calculator
    metrics_calculator = Metrics(data_loader.y_min, data_loader.y_max)
    
    # Calculate metrics (without verbose output)
    metrics_results = metrics_calculator.calculate_metrics(test_potential_y, test_y_hat, verbose=False)
    
    # Print results once
    print('\nResults:')
    print('-------------------')
    print(f'PEHE: {metrics_results["PEHE"]["value"]:.4f} ± '
          f'{metrics_results["PEHE"]["confidence_interval"]:.4f}')
    print(f'ATE: {metrics_results["ATE"]["value"]:.4f} ± '
          f'{metrics_results["ATE"]["confidence_interval"]:.4f}')
    print(f'MSE: {metrics_results["MSE"]:.4f}')
    print(f'RMSE: {metrics_results["RMSE"]:.4f}')
    print(f'MAE: {metrics_results["MAE"]:.4f}')
    
    # Save detailed results
    with open(f"results/{args.name}/results.txt", "w") as f:
        f.write(str(metrics_results))
    
    return test_y_hat, metrics_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Add command line arguments
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