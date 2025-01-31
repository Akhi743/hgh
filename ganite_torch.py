import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Generator, self).__init__()
        self.flag_dropout = flag_dropout
        
        self.fc1 = nn.Linear(input_dim + 2, h_dim)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        
        # Linear output for proper scaling
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)
        
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t, y):
        inputs = torch.cat([x, t, y], dim=1)
        
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))
        
        h31 = torch.relu(self.fc31(h2))
        y0 = self.fc32(h31)
        
        h41 = torch.relu(self.fc41(h2))
        y1 = self.fc42(h41)
        
        # Ensure outputs are in [0,1] range
        y0 = torch.clamp(y0, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        
        return torch.cat([y0, y1], dim=1)

class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Discriminator, self).__init__()
        self.flag_dropout = flag_dropout
        
        self.fc1 = nn.Linear(input_dim + 2, h_dim)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(h_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t, y, y_hat):
        input0 = (1. - t) * y + t * y_hat[:, 0].unsqueeze(1)
        input1 = t * y + (1. - t) * y_hat[:, 1].unsqueeze(1)
        inputs = torch.cat([x, input0, input1], dim=1)
        
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))
        
        return self.fc3(h2)

class InferenceNet(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(InferenceNet, self).__init__()
        self.flag_dropout = flag_dropout
        
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        
        # Linear output for proper scaling
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)
        
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(x)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(x))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))
        
        h31 = torch.relu(self.fc31(h2))
        y0 = self.fc32(h31)
        
        h41 = torch.relu(self.fc41(h2))
        y1 = self.fc42(h41)
        
        # Ensure outputs are in [0,1] range
        y0 = torch.clamp(y0, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        
        return torch.cat([y0, y1], dim=1)

def ganite_torch(train_x, train_t, train_y, test_x, train_potential_y, test_potential_y,
                parameters, name, flags):
    """GANITE implementation with proper output scaling."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Unpack parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    alpha = parameters['alpha']
    beta = parameters['beta']
    lr = parameters['lr']
    
    # Convert to tensors
    train_x = torch.FloatTensor(train_x).to(device)
    train_t = torch.FloatTensor(train_t).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    
    # Create dataloader
    train_dataset = TensorDataset(train_x, train_t.unsqueeze(1), train_y.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(train_x.shape[1], h_dim, flags['dropout']).to(device)
    discriminator = Discriminator(train_x.shape[1], h_dim, flags['dropout']).to(device)
    inference_net = InferenceNet(train_x.shape[1], h_dim, flags['dropout']).to(device)
    
    # Initialize optimizers
    if flags['adamw']:
        G_optimizer = optim.AdamW(generator.parameters(), lr=lr)
        D_optimizer = optim.AdamW(discriminator.parameters(), lr=lr)
        I_optimizer = optim.AdamW(inference_net.parameters(), lr=lr)
    else:
        G_optimizer = optim.Adam(generator.parameters(), lr=lr)
        D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
        I_optimizer = optim.Adam(inference_net.parameters(), lr=lr)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(f"results/{name}/logs"))
    model_dir = os.path.join(f"results/{name}/models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    with tqdm(range(iterations)) as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            g_losses, d_losses, i_losses = [], [], []
            
            for x, t, y in train_loader:
                # Train Discriminator
                discriminator.train()
                generator.eval()
                inference_net.eval()
                
                for _ in range(2):
                    y_tilde = generator(x, t, y)
                    d_logit = discriminator(x, t, y, y_tilde)
                    D_loss = nn.BCEWithLogitsLoss()(d_logit, t)
                    
                    D_optimizer.zero_grad()
                    D_loss.backward()
                    D_optimizer.step()
                
                # Train Generator
                generator.train()
                discriminator.eval()
                inference_net.eval()
                
                y_tilde = generator(x, t, y)
                d_logit = discriminator(x, t, y, y_tilde)
                G_loss_GAN = -nn.BCEWithLogitsLoss()(d_logit, t)
                
                # Factual loss with proper scaling
                y_est = torch.where(
                    t == 1,
                    y_tilde[:, 1].unsqueeze(1),
                    y_tilde[:, 0].unsqueeze(1)
                )
                G_loss_factual = nn.MSELoss()(y_est, y)
                G_loss = G_loss_factual + alpha * G_loss_GAN
                
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                
                # Train Inference Network
                inference_net.train()
                generator.eval()
                discriminator.eval()
                
                y_hat = inference_net(x)
                y_tilde = generator(x, t, y)
                
                # Properly handle counterfactual outcomes
                y_t0 = torch.where(
                    t == 1,
                    y_tilde[:, 0].unsqueeze(1),
                    y.clone()
                )
                y_t1 = torch.where(
                    t == 0,
                    y_tilde[:, 1].unsqueeze(1),
                    y.clone()
                )
                
                I_loss1 = nn.MSELoss()(y_hat[:, 0].unsqueeze(1), y_t0)
                I_loss2 = nn.MSELoss()(y_hat[:, 1].unsqueeze(1), y_t1)
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
            
            pbar.set_postfix({
                'G_loss': f'{np.mean(g_losses):.4f}',
                'D_loss': f'{np.mean(d_losses):.4f}',
                'I_loss': f'{np.mean(i_losses):.4f}'
            })
            
            # Save checkpoints
            if epoch % 1000 == 0 and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'inference_net_state_dict': inference_net.state_dict(),
                    'parameters': parameters
                }, os.path.join(model_dir, f'checkpoint_{epoch}.pt'))
    
    # Generate predictions with diagnostics
    inference_net.eval()
    with torch.no_grad():
        test_y_hat = inference_net(test_x).cpu().numpy()
        
        print("\n=== Model Output Analysis ===")
        print("Network predictions (before any post-processing):")
        print(f"Range: [{np.min(test_y_hat):.4f}, {np.max(test_y_hat):.4f}]")
        print(f"Mean: {np.mean(test_y_hat):.4f}")
        print(f"Std: {np.std(test_y_hat):.4f}")
        print("\nPer-outcome statistics:")
        print(f"Control (y0) - Range: [{np.min(test_y_hat[:,0]):.4f}, {np.max(test_y_hat[:,0]):.4f}]")
        print(f"Control (y0) - Mean: {np.mean(test_y_hat[:,0]):.4f}")
        print(f"Control (y0) - Std: {np.std(test_y_hat[:,0]):.4f}")
        print(f"Treated (y1) - Range: [{np.min(test_y_hat[:,1]):.4f}, {np.max(test_y_hat[:,1]):.4f}]")
        print(f"Treated (y1) - Mean: {np.mean(test_y_hat[:,1]):.4f}")
        print(f"Treated (y1) - Std: {np.std(test_y_hat[:,1]):.4f}")
    
    # Save final model
    torch.save({
        'epoch': iterations,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'inference_net_state_dict': inference_net.state_dict(),
        'parameters': parameters
    }, os.path.join(model_dir, 'final_model.pt'))
    
    writer.close()
    return test_y_hat