import numpy as np
from scipy.integrate import odeint

# Configuration
N_train = 1000  # Number of training samples
N_test = 100    # Number of test samples
Nx = 100        # Number of spatial points for solver
Nt = 100        # Number of temporal points for solver
P = 100         # Number of sensor points (input dimension for Branch net)

# Domain
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, 1, Nt)
nu = 0.01 / np.pi

def solve_burgers(u0, x, t, nu):
    """
    Solves Burgers equation using finite difference in space and odeint in time.
    u_t = nu * u_xx - u * u_x
    """
    dx = x[1] - x[0]
    
    def ode_system(u, t):
        # Boundary conditions are u=0 at ends, so we only solve for inner points
        # But for simplicity in indexing, let's keep full array and enforce BCs
        
        # Central difference for u_x
        # u_x[i] = (u[i+1] - u[i-1]) / (2*dx)
        
        # Central difference for u_xx
        # u_xx[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dx^2)
        
        u_xx = np.zeros_like(u)
        u_x = np.zeros_like(u)
        
        # Inner points
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        
        du_dt = nu * u_xx - u * u_x
        
        # Enforce BCs (Dirichlet u=0)
        du_dt[0] = 0
        du_dt[-1] = 0
        
        return du_dt

    # Solve
    u = odeint(ode_system, u0, t)
    return u

def generate_grf(N_samples, Nx, length_scale=0.2):
    """
    Generates Gaussian Random Fields for initial conditions.
    """
    X1, X2 = np.meshgrid(x, x)
    K = np.exp(-0.5 * (X1 - X2)**2 / length_scale**2)
    L = np.linalg.cholesky(K + 1e-13 * np.eye(Nx))
    
    u0_samples = []
    for _ in range(N_samples):
        u = np.dot(L, np.random.randn(Nx))
        # Enforce BCs for the initial condition smoothly
        # u(-1)=0, u(1)=0. A simple way is to multiply by a window function or just force it.
        # But GRF might not be 0 at boundaries. 
        # Let's enforce zero boundaries by subtracting linear interpolation or multiplying by (1-x^2)
        # Multiplying by (1-x^2) preserves the structure inside but forces 0 at ends.
        # Actually, for standard benchmarks, often Periodic BCs are used, but here it's Dirichlet.
        # Let's just force endpoints to 0. 
        # Better: u = u * (1 - x**2) might be too aggressive? 
        # Let's just set u[0]=0, u[-1]=0 and smooth nearby?
        # A clean way is to generate on inner points or use a sine expansion.
        # For simplicity and robustness, let's multiply by a smooth window.
        # But wait, the problem statement specific IC is -sin(pi*x), which is 0 at boundaries.
        # So we want functions that are 0 at boundaries.
        
        # Using a sum of sin bases is often better for Dirichlet BCs.
        # u0(x) = sum(a_k * sin(k * pi * (x+1)/2))
        
        temp = np.zeros(Nx)
        # Random coefficients for first N modes
        for k in range(1, 15):
             coef = np.random.randn() * (k**(-2)) # Decay
             temp += coef * np.sin(k * np.pi * (x + 1) / 2)
        
        # Normalize to have reasonable magnitude (e.g. max value ~1)
        if np.max(np.abs(temp)) > 0:
            temp = temp / np.max(np.abs(temp))
            
        u0_samples.append(temp)
        
    return np.array(u0_samples)

def generate_data():
    print("Generating training data...")
    # Generate random ICs
    u0_train = generate_grf(N_train, Nx)
    u0_test = generate_grf(N_test, Nx)
    
    # Add the specific case to test set or keep separate?
    # The user wants to "inference solve a specific burgers equation". 
    # I'll save the specific case separately.
    u0_specific = -np.sin(np.pi * x)
    
    # Solve for all
    u_train = []
    for i in range(N_train):
        sol = solve_burgers(u0_train[i], x, t, nu)
        u_train.append(sol)
        if (i+1) % 100 == 0:
            print(f"Solved {i+1}/{N_train}")
    
    u_test = []
    for i in range(N_test):
        sol = solve_burgers(u0_test[i], x, t, nu)
        u_test.append(sol)
        
    u_specific = solve_burgers(u0_specific, x, t, nu)
    
    # Format data for DeepONet
    # Branch input: u0 (sampled at P sensors). Here P=Nx is fine, or we can subsample.
    # Trunk input: (x, t) coordinates.
    # Output: u(x,t)
    
    # We save the raw arrays, processing can happen in train.py
    np.savez("burgers_data.npz", 
             x=x, t=t, 
             u0_train=u0_train, u_train=np.array(u_train),
             u0_test=u0_test, u_test=np.array(u_test),
             u0_specific=u0_specific, u_specific=u_specific)
    print("Data saved to burgers_data.npz")

if __name__ == "__main__":
    generate_data()

