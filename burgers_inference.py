import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_train import DeepONet # Import the model class

def inference():
    if not os.path.exists("deeponet_burgers.pth"):
        print("Model file not found. Run burgers_train.py first.")
        return
    if not os.path.exists("burgers_data.npz"):
        print("Data file not found. Run burgers_data_gen.py first.")
        return

    device = torch.device("cpu") # Inference on CPU is fine usually
    
    # Load data
    data = np.load("burgers_data.npz")
    u0_specific = data['u0_specific'] # (Nx,)
    u_specific_true = data['u_specific'] # (Nt, Nx)
    x = data['x']
    t = data['t']
    
    # Prepare inputs
    u0_tensor = torch.tensor(u0_specific, dtype=torch.float32).unsqueeze(0).to(device) # (1, P)
    
    X, T_mesh = np.meshgrid(x, t)
    xt_grid = np.column_stack((X.flatten(), T_mesh.flatten()))
    xt_tensor = torch.tensor(xt_grid, dtype=torch.float32).to(device) # (M, 2)
    
    # Load model
    model = DeepONet(input_branch=len(u0_specific), input_trunk=2, output_neurons=100).to(device)
    model.load_state_dict(torch.load("deeponet_burgers.pth", map_location=device))
    model.eval()
    
    # Predict
    with torch.no_grad():
        B = model.branch_net(u0_tensor) # (1, p)
        T = model.trunk_net(xt_tensor)   # (M, p)
        pred = torch.matmul(B, T.T) + model.bias # (1, M)
        
    u_pred = pred.numpy().reshape(len(t), len(x))
    
    # Calculate Error
    error = np.abs(u_specific_true - u_pred)
    mse = np.mean(error**2)
    print(f"Mean Squared Error on specific test case: {mse:.6e}")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Exact
    c1 = axes[0].contourf(t, x, u_specific_true.T, 100, cmap='jet')
    axes[0].set_title("Exact Solution")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    plt.colorbar(c1, ax=axes[0])
    
    # Predicted
    c2 = axes[1].contourf(t, x, u_pred.T, 100, cmap='jet')
    axes[1].set_title("Predicted Solution")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    plt.colorbar(c2, ax=axes[1])
    
    # Error
    c3 = axes[2].contourf(t, x, error.T, 100, cmap='jet')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    plt.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("burgers_result.png")
    print("Results saved to burgers_result.png")

    # Also plot snapshots
    plt.figure(figsize=(10, 6))
    times = [0.2, 0.5, 0.8]
    for i, ti in enumerate(times):
        t_idx = int(ti * len(t))
        plt.plot(x, u_specific_true[t_idx, :], 'b-', label=f"Exact t={ti}" if i==0 else None)
        plt.plot(x, u_pred[t_idx, :], 'r--', label=f"Pred t={ti}" if i==0 else None)
    
    plt.legend()
    plt.title("Snapshots at t=0.2, 0.5, 0.8")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.savefig("burgers_snapshots.png")
    print("Snapshots saved to burgers_snapshots.png")

if __name__ == "__main__":
    inference()

