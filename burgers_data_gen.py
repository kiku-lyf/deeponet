import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.fftpack import dct, idct

# Configuration
N_train = 1000
N_test = 100
Nx = 100        # Output spatial resolution
Nt = 100        # Output temporal resolution
Nx_fine = 256   # Internal resolution for spectral accuracy
nu = 0.01 / np.pi

# Domain
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, 1, Nt)

# Fine grid for internal calculation
x_fine = np.linspace(-1, 1, Nx_fine)
dx_fine = x_fine[1] - x_fine[0]

def solve_burgers_colehopf(u0_func_vals, x_in, t_out, nu):
    """
    Solves Burgers equation using Cole-Hopf transformation.
    u0_func_vals: initial condition values on x_in
    x_in: spatial grid
    t_out: time points to evaluate
    nu: viscosity
    """
    # 1. Integrate u0 to get potential U0
    # U0(x) = int_{-1}^x u0(z) dz
    U0 = cumtrapz(u0_func_vals, x_in, initial=0)
    
    # 2. Transform to Heat equation variable phi
    # phi = exp( - U / (2*nu) )
    # To avoid overflow/underflow, we can shift U by a constant? 
    # u = -2nu phi_x / phi. Scaling phi by constant C doesn't change u.
    # So we can subtract min(U0) or something.
    U0_shift = U0 - np.min(U0) # Shift so min is 0 (max phi is 1)
    # Actually if U0 is large positive, exp(-large) -> 0 (underflow).
    # If U0 is large negative, exp(large) -> inf (overflow).
    # nu is small (~0.003). 1/(2nu) ~ 150.
    # If U0 range is large, we have issues.
    # With GRF normalized to 1, U0 range is roughly [-1, 1]*2 = 2?
    # 150*2 = 300. exp(300) is fine. exp(-300) is 1e-130 (fine).
    # We should center U0 to avoid one extreme.
    phi0 = np.exp(-U0 / (2 * nu))
    
    # 3. Solve Heat Equation phi_t = nu * phi_xx using DCT (Cosine Transform)
    # Neumann BCs correspond to DCT Type 2 (standard scipy dct)
    # Expansion: phi(x) = sum a_k cos(k * pi * (x+1)/2)
    # Wait, DCT assumes grid indices 0..N-1.
    # Standard DCT-II implies even extension at boundaries?
    # Yes, DCT-II matches Neumann BCs at both ends.
    
    phi0_hat = dct(phi0, type=2, norm='ortho')
    
    # Wavenumbers
    # For DCT-II on N points, modes are k=0, 1, ..., N-1
    # Eigenvalues for Heat eq on [-1, 1] (Length L=2)
    # lambda_k = (k * pi / L)^2 = (k * pi / 2)^2
    k = np.arange(Nx_fine)
    lambda_k = (k * np.pi / 2)**2
    
    phi_sol = []
    
    # Precompute time evolution factors
    # shape (Nt, Nx)
    # We need to evaluate at specific t
    
    for t_val in t_out:
        decay = np.exp(-nu * lambda_k * t_val)
        phi_hat_t = phi0_hat * decay
        phi_t = idct(phi_hat_t, type=2, norm='ortho')
        
        # 4. Recover u = -2 * nu * phi_x / phi
        # Compute phi_x. Can use spectral derivative or finite difference.
        # Spectral derivative for Cosine series -> Sine series.
        # phi(x) = sum a_k cos(k theta). phi' = sum -a_k * k * sin(k theta) * dtheta/dx
        # But simpler to just use central difference on fine grid since Nx_fine is 256
        # Or even better, use complex step or just finite diff.
        
        phi_x = np.zeros_like(phi_t)
        # Central difference
        phi_x[1:-1] = (phi_t[2:] - phi_t[:-2]) / (2 * dx_fine)
        # Boundaries (Neumann BCs imply phi_x = 0)
        phi_x[0] = 0
        phi_x[-1] = 0
        
        # Avoid division by zero
        # phi is strictly positive if started positive? Yes, heat equation preserves positivity.
        u_t = -2 * nu * phi_x / (phi_t + 1e-30)
        
        # Enforce u=0 at boundaries explicitly (though phi_x=0 implies it)
        u_t[0] = 0
        u_t[-1] = 0
        
        phi_sol.append(u_t)
        
    phi_sol = np.array(phi_sol) # (Nt, Nx_fine)
    
    # Interpolate to output grid x
    u_interp = []
    for i in range(Nt):
        f = interp1d(x_fine, phi_sol[i], kind='cubic')
        u_interp.append(f(x))
        
    return np.array(u_interp)

def generate_grf(N_samples, Nx):
    """
    Generates random initial conditions satisfying u(-1)=u(1)=0.
    Using Sine series expansion to guarantee Dirichlet BCs.
    u(x) = sum_{k=1}^K a_k sin(k * pi * (x+1)/2)
    """
    u0_samples = []
    
    # Coordinates for generation
    # We generate on fine grid to ensure integration accuracy in Cole-Hopf
    x_gen = x_fine
    
    for _ in range(N_samples):
        temp = np.zeros_like(x_gen)
        # Random coefficients with decay
        # K modes
        K = 20
        for k in range(1, K+1):
             # Coefficients normally distributed, decaying with k
             # Using k^-2 decay for smoothness
             coef = np.random.randn() * (k**(-2.0))
             temp += coef * np.sin(k * np.pi * (x_gen + 1) / 2)
        
        # Normalize max amplitude to range [0.5, 1.5] roughly to cover varied regimes
        max_val = np.max(np.abs(temp))
        if max_val > 0:
            target_amp = 1.0 # Standardize around 1
            temp = temp / max_val * target_amp
            
        u0_samples.append(temp)
        
    return np.array(u0_samples)

def generate_data():
    print("Generating training data with Cole-Hopf solver...")
    
    # 1. Generate random ICs on FINE grid
    u0_train_fine = generate_grf(N_train, Nx_fine)
    u0_test_fine = generate_grf(N_test, Nx_fine)
    
    # 2. Specific case: -sin(pi*x)
    u0_specific_fine = -np.sin(np.pi * x_fine)
    
    # 3. Solve
    u_train = []
    for i in range(N_train):
        sol = solve_burgers_colehopf(u0_train_fine[i], x_fine, t, nu)
        u_train.append(sol)
        if (i+1) % 100 == 0:
            print(f"Solved {i+1}/{N_train}")
            
    u_test = []
    for i in range(N_test):
        sol = solve_burgers_colehopf(u0_test_fine[i], x_fine, t, nu)
        u_test.append(sol)
        
    u_specific = solve_burgers_colehopf(u0_specific_fine, x_fine, t, nu)
    
    # 4. Downsample/Interpolate ICs to output grid x for saving
    # The network takes u0 sampled at P sensors (x grid)
    # u_train is already on x grid (from solve function return)
    
    # Interpolate u0 to x
    u0_train = []
    for i in range(N_train):
        f = interp1d(x_fine, u0_train_fine[i], kind='cubic')
        u0_train.append(f(x))
    u0_train = np.array(u0_train)
        
    u0_test = []
    for i in range(N_test):
        f = interp1d(x_fine, u0_test_fine[i], kind='cubic')
        u0_test.append(f(x))
    u0_test = np.array(u0_test)
        
    f_spec = interp1d(x_fine, u0_specific_fine, kind='cubic')
    u0_specific = f_spec(x)
    
    # Check for NaNs or Infs
    if np.any(np.isnan(u_train)) or np.any(np.isinf(u_train)):
        print("WARNING: NaNs or Infs detected in training data!")
    else:
        print(f"Data statistics: Min {np.min(u_train):.4f}, Max {np.max(u_train):.4f}")
    
    np.savez("burgers_data.npz", 
             x=x, t=t, 
             u0_train=u0_train, u_train=np.array(u_train),
             u0_test=u0_test, u_test=np.array(u_test),
             u0_specific=u0_specific, u_specific=u_specific)
    print("Data saved to burgers_data.npz")

if __name__ == "__main__":
    generate_data()
