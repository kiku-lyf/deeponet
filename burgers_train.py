import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Configuration
BATCH_SIZE = 1000
EPOCHS = 5000 # Can be adjusted
LEARNING_RATE = 1e-3
P_SENSORS = 100 # Match P in data gen
P_HIDDEN = 100 # Output size of branch/trunk

class DeepONet(nn.Module):
    def __init__(self, input_branch, input_trunk, output_neurons=100):
        super(DeepONet, self).__init__()
        
        self.branch_net = nn.Sequential(
            nn.Linear(input_branch, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_neurons),
        )
        
        self.trunk_net = nn.Sequential(
            nn.Linear(input_trunk, 128),
            nn.Tanh(), # Tanh often works better for trunk net (coordinates)
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_neurons),
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, xt):
        # u0: (batch_size, input_branch)
        # xt: (batch_size, input_trunk)
        
        B = self.branch_net(u0) # (batch, p)
        T = self.trunk_net(xt)  # (batch, p)
        
        # Output is dot product of B and T
        output = torch.sum(B * T, dim=1, keepdim=True) + self.bias
        return output

def load_data():
    data = np.load("burgers_data.npz")
    
    u0_train = data['u0_train'] # (N_train, P)
    u_train = data['u_train']   # (N_train, Nt, Nx)
    x = data['x']
    t = data['t']
    
    # We need to reshape data into pairs of (u0, (x,t)) -> u
    # Since DeepONet takes one u0 and one (x,t) to produce one u(x,t)
    # But usually we train on batches.
    # We can repeat u0 for all grid points or sample grid points.
    # For full grid training:
    # Inputs: 
    #   Branch: u0 repeated Nt*Nx times per sample
    #   Trunk: meshgrid (x,t) flattened
    # Targets: u flattened
    
    # Let's create a dataset where for each u0, we have all (x,t) points
    # Actually, for efficiency, we often batch u0 and evaluate T on all points or batch points.
    # PyTorch implementation:
    # If we pass u0 (N, P) and xt (M, 2), we want output (N, M) if fully cross.
    # But standard NN takes (Batch, In) -> (Batch, Out).
    # So we flatten.
    
    X, T_mesh = np.meshgrid(x, t)
    xt_grid = np.column_stack((X.flatten(), T_mesh.flatten())) # (Nt*Nx, 2)
    
    # Repeat u0 and tile xt
    # This might be too large for memory if N_train is big.
    # N_train=1000, Nt*Nx = 100*100 = 10000. Total 10^7 samples.
    # That's manageable (10M samples).
    
    # Let's verify dimensions
    N_train = u0_train.shape[0]
    
    # Create tensors
    u0_train_torch = torch.tensor(u0_train, dtype=torch.float32)
    xt_grid_torch = torch.tensor(xt_grid, dtype=torch.float32)
    u_train_torch = torch.tensor(u_train.reshape(N_train, -1), dtype=torch.float32) # (N_train, Nt*Nx)
    
    return u0_train_torch, xt_grid_torch, u_train_torch, x, t

def train():
    if not os.path.exists("burgers_data.npz"):
        print("Data file not found. Run burgers_data_gen.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    u0_train, xt_grid, u_train, _, _ = load_data()
    u0_train = u0_train.to(device)
    xt_grid = xt_grid.to(device) # (M, 2)
    u_train = u_train.to(device)   # (N, M)
    
    # Model
    model = DeepONet(input_branch=P_SENSORS, input_trunk=2, output_neurons=P_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training Loop
    # We can batch over N_train (different functions)
    # The trunk input is fixed (the grid), so we can compute T once per forward if we passed all xt?
    # No, DeepONet usually:
    # Option 1: Forward (batch_u0, batch_xt) -> batch_out. 
    #           Here we need to pair specific u0 with specific xt.
    # Option 2: Efficient implementation where T = trunk(all_xt) computed once?
    #           Only if we train on full grid for every u0.
    #           Here u_train is (N, M).
    #           B = branch(u0) -> (N, p)
    #           T = trunk(xt_grid) -> (M, p)
    #           Pred = B @ T.T + bias -> (N, M)
    #           Loss = MSE(Pred, u_train)
    # This is efficient and feasible for 10M elements on GPU/CPU.
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Branch output
        B = model.branch_net(u0_train) # (N, p)
        
        # Trunk output
        T = model.trunk_net(xt_grid)   # (M, p)
        
        # Prediction
        # (N, p) x (p, M) -> (N, M)
        pred = torch.matmul(B, T.T) + model.bias
        
        loss = criterion(pred, u_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")
            
    # Save model
    torch.save(model.state_dict(), "deeponet_burgers.pth")
    print("Model saved to deeponet_burgers.pth")

if __name__ == "__main__":
    train()

