import numpy as np
import scipy.io as sio

def sample_grf_periodic(s, gamma=4.0, tau=5.0, sigma=25.0, rng=None):
    """
    Sample a real-valued periodic Gaussian random field on [0,1)
    with covariance operator C = sigma^2 (-Δ + tau^2 I)^(-gamma).
    Returns u0 on a uniform grid of size s.
    """
    if rng is None:
        rng = np.random.default_rng()

    # grid
    x = np.linspace(0.0, 1.0, s, endpoint=False)

    # Fourier wavenumbers (cycles per unit) for FFT on [0,1)
    k = np.fft.fftfreq(s, d=1.0/s)  # ..., -2, -1, 0, 1, 2, ...
    omega = 2.0 * np.pi * k         # angular wavenumbers

    # spectral density ~ ((omega^2 + tau^2)^(-gamma))
    # amplitude std for complex coeffs: sqrt(S(k))
    Sk = (omega**2 + tau**2) ** (-gamma)
    Sk[0] = (tau**2) ** (-gamma)    # safe for k=0

    # build complex Fourier coefficients with conjugate symmetry -> real field
    # We'll sample half-spectrum and mirror.
    uhat = np.zeros(s, dtype=np.complex128)

    # k=0 real
    uhat[0] = rng.normal() * np.sqrt(Sk[0])

    # if s even, Nyquist mode k = s/2 is real
    if s % 2 == 0:
        nyq = s // 2
        uhat[nyq] = rng.normal() * np.sqrt(Sk[nyq])

        pos = np.arange(1, nyq)
    else:
        pos = np.arange(1, (s+1)//2)

    # positive modes: complex normal
    a = rng.normal(size=pos.size)
    b = rng.normal(size=pos.size)
    uhat[pos] = (a + 1j*b) * np.sqrt(Sk[pos] / 2.0)

    # negative modes: conjugate
    uhat[-pos] = np.conj(uhat[pos])

    # scale by sigma
    uhat *= sigma

    # inverse FFT to real space
    u0 = np.fft.ifft(uhat).real

    # optional shift like chebfun uu(t-0.5): a spatial shift of 0.5
    # shift by half-domain in physical space = phase factor in Fourier space
    # uncomment if you want closer to their exact shift behavior:
    # shift = 0.5
    # uhat_shift = uhat * np.exp(-1j * omega * shift)
    # u0 = np.fft.ifft(uhat_shift).real

    return x, u0


def burgers_etdrk4(u0, visc=0.01, T=1.0, dt=2e-4, dealias=True):
    """
    Solve u_t + u u_x = visc u_xx on periodic [0,1) using Fourier spectral + ETDRK4.
    u0: array shape (s,)
    Returns u at times t_out provided externally by sampling snapshots.
    """
    u = u0.copy()
    s = u.size
    k = np.fft.fftfreq(s, d=1.0/s)
    omega = 2.0*np.pi*k
    L = -visc * (omega**2)  # in Fourier, u_xx -> -(omega^2) uhat, so visc u_xx -> -visc*omega^2

    # ETDRK4 coefficients (Kassam & Trefethen 2005)
    E  = np.exp(L*dt)
    E2 = np.exp(L*dt/2.0)

    # contour integral approximation
    M = 32
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5)/M)
    LR = L[:, None]*dt + r[None, :]
    Q  = dt*np.real(np.mean((np.exp(LR/2.0)-1.0)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4.0 - LR + np.exp(LR)*(4.0 - 3.0*LR + LR**2)) / (LR**3), axis=1))
    f2 = dt*np.real(np.mean(( 2.0 + LR + np.exp(LR)*(-2.0 + LR)) / (LR**3), axis=1))
    f3 = dt*np.real(np.mean((-4.0 - 3.0*LR - LR**2 + np.exp(LR)*(4.0 - LR)) / (LR**3), axis=1))

    # dealias mask (2/3 rule)
    if dealias:
        kmax = np.max(np.abs(k))
        cutoff = (2.0/3.0) * kmax
        mask = (np.abs(k) <= cutoff).astype(float)
    else:
        mask = np.ones_like(k)

    def nonlinear_term(u_phys):
        # N(u) = -0.5 (u^2)_x
        u2 = u_phys*u_phys
        u2_hat = np.fft.fft(u2) * mask
        du2dx = np.fft.ifft(1j*omega*u2_hat).real
        return -0.5 * du2dx

    nsteps = int(np.round(T/dt))
    # store all time steps? not needed
    uhat = np.fft.fft(u)

    for _ in range(nsteps):
        u = np.fft.ifft(uhat).real
        Nv = np.fft.fft(nonlinear_term(u)) * mask

        a = E2*uhat + Q*Nv
        ua = np.fft.ifft(a).real
        Na = np.fft.fft(nonlinear_term(ua)) * mask

        b = E2*uhat + Q*Na
        ub = np.fft.ifft(b).real
        Nb = np.fft.fft(nonlinear_term(ub)) * mask

        c = E2*a + Q*(2*Nb - Nv)
        uc = np.fft.ifft(c).real
        Nc = np.fft.fft(nonlinear_term(uc)) * mask

        uhat = E*uhat + f1*Nv + f2*(Na + Nb) + f3*Nc

    return np.fft.ifft(uhat).real


def generate_burger_mat(
    mat_path="Burger.mat",
    N=10,
    gamma=4.0,
    tau=5.0,
    sigma=25.0,   # 注意：你matlab里 sigma = 25^(2) 写成了 625；那里把 sigma 当成“方差尺度”用的写法有点混。
    visc=0.01,
    s=4096,
    steps=100,
    nn=101,
    dt_internal=2e-4,
    seed=0
):
    rng = np.random.default_rng(seed)

    tspan = np.linspace(0.0, 1.0, steps+1)          # 101
    X = np.linspace(0.0, 1.0, nn)                    # 101 sensors

    input_arr  = np.zeros((N, nn), dtype=np.float64)
    output_arr = np.zeros((N, steps+1, nn), dtype=np.float64)

    # helper: sample u(x) on sensors X from grid values on uniform grid
    # since grid is uniform periodic, we can do interpolation via FFT or simple linear interp.
    # For simplicity: use np.interp with periodic wrap (good enough for nn=101)
    def sample_on_X(x_grid, u_grid):
        # x_grid in [0,1) endpoint=False, X includes 1.0
        # map X==1.0 back to 0.0
        Xp = X.copy()
        Xp[-1] = 0.0
        return np.interp(Xp, x_grid, u_grid, period=1.0)

    # We need outputs at each tspan[k]. We'll integrate piecewise between snapshots.
    # Use internal dt_internal for accuracy.
    dt = dt_internal
    chunk_T = 1.0/steps
    # ensure chunk_T is multiple of dt
    n_sub = int(np.round(chunk_T/dt))
    dt = chunk_T / n_sub  # adjust slightly to fit exactly

    for j in range(N):
        xg, u0 = sample_grf_periodic(s, gamma=gamma, tau=tau, sigma=sigma, rng=rng)

        # record input at sensors
        input_arr[j, :] = sample_on_X(xg, u0)

        # evolve and record outputs at each snapshot
        u = u0.copy()
        output_arr[j, 0, :] = input_arr[j, :]  # t=0

        for k in range(1, steps+1):
            # integrate from t_{k-1} to t_k in n_sub ETDRK4 steps
            # We'll call burgers_etdrk4 for one chunk.
            u = burgers_etdrk4(u, visc=visc, T=chunk_T, dt=dt, dealias=True)
            output_arr[j, k, :] = sample_on_X(xg, u)

        print(f"generated {j+1}/{N}")

    # save .mat
    sio.savemat(mat_path, {
        "input": input_arr,
        "output": output_arr,
        "tspan": tspan,
        "gamma": gamma,
        "tau": tau,
        "sigma": sigma
    })
    print(f"Saved {mat_path}: input {input_arr.shape}, output {output_arr.shape}, tspan {tspan.shape}")


if __name__ == "__main__":
    # 重要：如果你想更贴近你matlab脚本里的写法，sigma 你可能需要传 625（=25^2）
    generate_burger_mat(
        mat_path="Burger.mat",
        N=1000,
        gamma=4.0,
        tau=5.0,
        sigma=625.0,     # 对齐你 matlab: sigma = 25^(2)
        visc=0.01,
        s=4096,
        steps=100,
        nn=101,
        dt_internal=2e-4,
        seed=0
    )
