import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Configuration
BATCH_SIZE = 64
EPOCHS = 5000
LEARNING_RATE = 1e-3
P_SENSORS = 100 # Match P in data gen (Nx)
P_HIDDEN = 128  # Width of hidden layers

class DeepONet(nn.Module):
    def __init__(self, input_branch, input_trunk, output_neurons=100):
        super(DeepONet, self).__init__()
        
        self.branch_net = nn.Sequential(
            nn.Linear(input_branch, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_neurons),
        )
        
        self.trunk_net = nn.Sequential(
            nn.Linear(input_trunk, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_neurons),
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, xt):
        # u0: (batch_size, input_branch)
        # xt: (batch_size or global, input_trunk)
        
        B = self.branch_net(u0) # (batch, p)
        T = self.trunk_net(xt)  # (M, p) or (batch, p)
        
        # If xt is the full grid (M points) and u0 is a batch (N samples)
        # We want output (N, M)
        # B: (N, p)
        # T: (M, p)
        # Out = B @ T.T -> (N, M)
        
        output = torch.matmul(B, T.T) + self.bias
        return output

def load_data():
    data = np.load("burgers_data.npz")
    
    u0_train = data['u0_train'] # (N_train, P)
    u_train = data['u_train']   # (N_train, Nt, Nx) -> flatten spatial/temporal
    x = data['x']
    t = data['t']
    
    N_train = u0_train.shape[0]
    
    # Create meshgrid for Trunk
    X, T_mesh = np.meshgrid(x, t)
    xt_grid = np.column_stack((X.flatten(), T_mesh.flatten())) # (Nt*Nx, 2)
    
    # Flatten u_train to (N_train, Nt*Nx)
    u_train_flat = u_train.reshape(N_train, -1)
    
    # Convert to Tensor
    u0_train_torch = torch.tensor(u0_train, dtype=torch.float32)
    xt_grid_torch = torch.tensor(xt_grid, dtype=torch.float32)
    u_train_torch = torch.tensor(u_train_flat, dtype=torch.float32)
    
    return u0_train_torch, xt_grid_torch, u_train_torch

def train():
    if not os.path.exists("burgers_data.npz"):
        print("Data file not found. Run burgers_data_gen.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    u0_full, xt_grid, u_full = load_data()
    
    # Move fixed trunk input to device
    xt_grid = xt_grid.to(device)
    
    # Create DataLoader for Branch inputs and Targets
    dataset = TensorDataset(u0_full, u_full)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = DeepONet(input_branch=P_SENSORS, input_trunk=2, output_neurons=P_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_u0, batch_u in loader:
            batch_u0 = batch_u0.to(device)
            batch_u = batch_u.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # batch_u0: (batch, P)
            # xt_grid: (M, 2) -> Fixed for all samples
            pred = model(batch_u0, xt_grid) # (batch, M)
            
            loss = criterion(pred, batch_u)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
    # Save model
    torch.save(model.state_dict(), "deeponet_burgers.pth")
    print("Model saved to deeponet_burgers.pth")

if __name__ == "__main__":
    train()
