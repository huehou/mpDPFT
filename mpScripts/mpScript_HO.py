#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial

def psi_1d_sq(n, x):
    """
    1D HO probability density |φ_n(x)|^2 in HO units (ℏ=m=ω=1).
    """
    A = 1.0 / (np.pi**0.25 * np.sqrt(2**n * factorial(n)))
    return (A * eval_hermite(n, x) * np.exp(-x**2/2))**2

def shell_cut_density(n, xs, y0=0.0):
    """
    2D shell-n density along the cut y=y0:
    ρ_n(x,0) = sum_{k=0}^n |φ_k(x)|^2 * |φ_{n-k}(y0)|^2.
    """
    # precompute φ_{n-k}(y0)^2 for k=0..n
    psi_y0 = np.array([psi_1d_sq(n - k, np.array([y0]))[0] for k in range(n + 1)])
    # build the cut density
    dens = np.zeros_like(xs)
    for k in range(n + 1):
        dens += psi_1d_sq(k, xs) * psi_y0[k]
    return dens

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <N_particles>")
        sys.exit(1)

    Np = int(sys.argv[1])

    # figure out how many full shells to fill
    # total states up to shell m: T(m) = (m+1)*(m+2)/2
    m = 0
    while (m + 1) * (m + 2) // 2 < Np:
        m += 1

    # number of states in all shells < m:
    T_prev = m * (m + 1) // 2
    R = Np - T_prev
    f = min(R / (m + 1), 1.0)  # fractional fill of shell m

    # load the cut data (x, y)
    data = np.loadtxt('mpDPFT_CutData.dat')
    x_data, y_data = data[:, 0], data[:, 1]

    # build a smooth x-grid over the data range
    xs = np.linspace(x_data.min(), x_data.max(), 1000)

    # sum shell densities
    density = np.zeros_like(xs)
    for n in range(m):
        density += shell_cut_density(n, xs)

    # add fractional last shell if needed
    if f > 0:
        density += f * shell_cut_density(m, xs)

    # plotting
    plt.plot(xs, density, lw=2, label=f'2D HO cut density (N={Np})')
    plt.plot(x_data, y_data, 'k--', label='mpDPFT_CutData')
    plt.xlabel('$x$ (HO units)')
    plt.ylabel(r'$\rho(x,y=0)$')
    plt.legend()
    plt.tight_layout()

    # grab the current figure
    fig = plt.gcf()
    # on window-close, exit the script
    fig.canvas.mpl_connect('close_event', lambda event: sys.exit(0))

    plt.show()

if __name__ == '__main__':
    main()
