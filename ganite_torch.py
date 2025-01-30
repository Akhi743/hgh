import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from model import Generator, GeneratorDeep, Discriminator, InferenceNet, InferenceNetDeep
from utils import (parameter_setting_discriminator, parameter_setting_generator,
                  parameter_setting_inference_net, parameter_setting_test)
from metrics import PEHE, ATE  # Changed from metrics_all to metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ganite_torch(train_x, train_t, train_y, test_x, train_potential_y, test_potential_y, parameters, name, flags):
    """GANITE implementation in PyTorch.
    
    Args:
        train_x: training features
        train_t: training treatments
        train_y: training observed outcomes
        test_x: test features
        train_potential_y: training potential outcomes
        test_potential_y: test potential outcomes
        parameters: model parameters
        name: experiment name
        flags: additional flags for model configuration
    
    Returns:
        test_y_hat: predicted potential outcomes for test set
    """
    # Unpack parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    alpha = parameters['alpha']
    beta = parameters['beta']

    # Convert to PyTorch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_t = torch.tensor(train_t, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

    # Create dataloader
    train_dataset = TensorDataset(train_x, train_t, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = GeneratorDeep(train_x.shape[1], h_dim, flags['dropout'], 15).to(device)
    discriminator = Discriminator(train_x.shape[1], h_dim, flags['dropout']).to(device)
    inference_net = InferenceNetDeep(train_x.shape[1], h_dim, flags['dropout'], 15).to(device)

    # Initialize optimizers
    if flags['adamw']:
        G_optimizer = optim.AdamW(generator.parameters(), lr=parameters['lr'])
        D_optimizer = optim.AdamW(discriminator.parameters(), lr=parameters['lr'])
        I_optimizer = optim.AdamW(inference_net.parameters(), lr=parameters['lr'])
    else:
        G_optimizer = optim.Adam(generator.parameters(), lr=parameters['lr'])
        D_optimizer = optim.Adam(discriminator.parameters(), lr=parameters['lr'])
        I_optimizer = optim.Adam(inference_net.parameters(), lr=parameters['lr'])

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(f"results/{name}/logs"))
    model_dir = os.path.join(f"results/{name}/models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Training loop
    with tqdm(range(iterations)) as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            g_losses, d_losses, i_losses = [], [], []

            for x, t, y in train_loader:
                t = t.unsqueeze(1)
                y = y.unsqueeze(1)

                # Train Discriminator
                parameter_setting_discriminator(generator, discriminator, inference_net)
                for _ in range(2):
                    y_tilde = generator(x, t, y)
                    d_logit = discriminator(x, t, y, y_tilde)
                    D_loss = nn.BCEWithLogitsLoss()(d_logit, t)

                    D_optimizer.zero_grad()
                    D_loss.backward(retain_graph=True)
                    D_optimizer.step()

                # Train Generator
                parameter_setting_generator(generator, discriminator, inference_net)
                y_tilde = generator(x, t, y)
                d_logit = discriminator(x, t, y, y_tilde)
                G_loss_GAN = -nn.BCEWithLogitsLoss()(d_logit, t)
                
                y_est = t * y_tilde[:, 1].view(-1, 1) + (1 - t) * y_tilde[:, 0].view(-1, 1)
                G_loss_factual = nn.BCEWithLogitsLoss()(y_est, y)
                G_loss = G_loss_factual + alpha * G_loss_GAN

                G_optimizer.zero_grad()
                G_loss.backward(retain_graph=True)
                G_optimizer.step()

                # Train Inference Network
                parameter_setting_inference_net(generator, discriminator, inference_net)
                y_hat = inference_net(x)
                y_tilde = generator(x, t, y)

                y_t0 = t * y + (1 - t) * y_tilde[:, 1].view(-1, 1)
                I_loss1 = nn.BCEWithLogitsLoss()(y_hat[:, 1].view(-1, 1), y_t0)
                y_t1 = (1 - t) * y + t * y_tilde[:, 0].view(-1, 1)
                I_loss2 = nn.BCEWithLogitsLoss()(y_hat[:, 0].view(-1, 1), y_t1)

                I_loss = I_loss1 + I_loss2

                I_optimizer.zero_grad()
                I_loss.backward()
                I_optimizer.step()

                # Record losses
                g_losses.append(G_loss.item())
                d_losses.append(D_loss.item())
                i_losses.append(I_loss.item())

            # Log metrics
            writer.add_scalar('Loss/Generator', np.mean(g_losses), epoch)
            writer.add_scalar('Loss/Discriminator', np.mean(d_losses), epoch)
            writer.add_scalar('Loss/Inference', np.mean(i_losses), epoch)

            # Save models periodically
            if epoch % 1000 == 0 and epoch > 0:
                torch.save(generator.state_dict(), 
                         os.path.join(model_dir, f'generator_{epoch}.pt'))
                torch.save(discriminator.state_dict(), 
                         os.path.join(model_dir, f'discriminator_{epoch}.pt'))
                torch.save(inference_net.state_dict(), 
                         os.path.join(model_dir, f'inference_{epoch}.pt'))

    # Generate final predictions
    parameter_setting_test(generator, discriminator, inference_net)
    test_y_hat = inference_net(test_x).cpu().detach().numpy()
    
    return test_y_hat