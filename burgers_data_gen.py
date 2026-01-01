import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Configuration
N_train = 1000
N_test = 100
Nx = 100        # Output spatial resolution
Nt = 100        # Output temporal resolution
nu = 0.01 / np.pi

# Domain
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, 1, Nt)

def cheb_nodes(N):
    # Chebyshev nodes on [-1, 1]
    # x_k = -cos(k * pi / N), k=0..N
    return -np.cos(np.pi * np.arange(N+1) / N)

def solve_burgers_viscous(u0, x, t, nu):
    """
    Solves Burgers equation u_t + u u_x = nu u_xx using finite difference with upwinding.
    This is generally more stable than central difference for advection.
    """
    dx = x[1] - x[0]
    
    def ode_system(u, t):
        # u is (Nx,)
        # Enforce BCs: u[0]=0, u[-1]=0 (Dirichlet)
        
        # Derivatives
        u_x = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        
        # Central difference for Diffusion: nu * u_xx
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
        # Upwind scheme for Advection: - u * u_x
        # If u > 0, use backward difference (u[i] - u[i-1])/dx
        # If u < 0, use forward difference (u[i+1] - u[i])/dx
        
        # Vectorized upwind
        # Backward diff
        ux_b = (u[1:-1] - u[:-2]) / dx
        # Forward diff
        ux_f = (u[2:] - u[1:-1]) / dx
        
        # Choose based on velocity direction
        # We need u_x at i.
        # term is - u[i] * u_x[i]
        
        # Actually simpler is Conservative Form? (u^2/2)_x
        # But let's stick to simple upwind for stability.
        
        u_val = u[1:-1]
        u_x_val = np.zeros_like(u_val)
        
        mask_pos = u_val > 0
        mask_neg = ~mask_pos
        
        u_x_val[mask_pos] = ux_b[mask_pos]
        u_x_val[mask_neg] = ux_f[mask_neg]
        
        du_dt = np.zeros_like(u)
        du_dt[1:-1] = nu * u_xx[1:-1] - u_val * u_x_val
        
        # Boundary Conditions (fixed at 0)
        du_dt[0] = 0
        du_dt[-1] = 0
        
        return du_dt

    # Solve
    # Increase mxstep for stiff problems if needed, though nu is small so it's advection dominated
    u = odeint(ode_system, u0, t, rtol=1e-5, atol=1e-5)
    return u

def generate_grf(N_samples, Nx):
    """
    Generates random initial conditions.
    u0(x) = sum a_k sin(k pi (x+1)/2)
    """
    u0_samples = []
    
    for _ in range(N_samples):
        temp = np.zeros(Nx)
        # Random coefficients
        # Use only first few low frequency modes to ensure smoothness
        # High frequencies with low viscosity -> shock waves -> instability
        K = 10 
        for k in range(1, K+1):
             coef = np.random.randn() * (k**(-2.0))
             temp += coef * np.sin(k * np.pi * (x + 1) / 2)
        
        # Normalize to reasonable amplitude
        # If amplitude is too high, Reynolds number is huge -> shocks -> numerical explosion
        # Max u ~ 1.0 with nu ~ 0.003 -> Re ~ 300.
        # This is high but manageable with upwinding.
        # Let's keep it safe around 0.5 - 1.0
        max_val = np.max(np.abs(temp))
        if max_val > 0:
            temp = temp / max_val  # Normalize to 1
            
        u0_samples.append(temp)
        
    return np.array(u0_samples)

def generate_data():
    print("Generating training data using Upwind Finite Difference...")
    
    # 1. Generate random ICs
    u0_train = generate_grf(N_train, Nx)
    u0_test = generate_grf(N_test, Nx)
    u0_specific = -np.sin(np.pi * x)
    
    # 2. Solve
    # We solve on the grid directly.
    # To be safer, we could solve on a finer grid and downsample, 
    # but let's try standard grid first with stable solver.
    
    u_train = []
    for i in range(N_train):
        sol = solve_burgers_viscous(u0_train[i], x, t, nu)
        u_train.append(sol)
        if (i+1) % 100 == 0:
            print(f"Solved {i+1}/{N_train}")
            
    u_test = []
    for i in range(N_test):
        sol = solve_burgers_viscous(u0_test[i], x, t, nu)
        u_test.append(sol)
        
    u_specific = solve_burgers_viscous(u0_specific, x, t, nu)
    
    u_train = np.array(u_train)
    u_test = np.array(u_test)
    
    # Check Statistics
    print(f"Data statistics: Min {np.min(u_train):.4f}, Max {np.max(u_train):.4f}")
    if np.max(np.abs(u_train)) > 100:
        print("WARNING: Data explosion detected!")
    
    np.savez("burgers_data.npz", 
             x=x, t=t, 
             u0_train=u0_train, u_train=u_train,
             u0_test=u0_test, u_test=u_test,
             u0_specific=u0_specific, u_specific=u_specific)
    print("Data saved to burgers_data.npz")

if __name__ == "__main__":
    generate_data()
