#!/usr/bin/env python3
import sys
import glob
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
    psi_y0 = np.array([psi_1d_sq(n - k, np.array([y0]))[0] for k in range(n + 1)])
    dens = np.zeros_like(xs)
    for k in range(n + 1):
        dens += psi_1d_sq(k, xs) * psi_y0[k]
    return dens

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <N_particles>")
        sys.exit(1)

    Np = int(sys.argv[1])
    data_files = sorted(glob.glob('mpDPFT_*CutData.dat'))
    if not data_files:
        print("Error: No files found matching 'mpDPFT_*CutData.dat'")
        sys.exit(1)

    # determine shell fill
    m = 0
    while (m + 1) * (m + 2) // 2 < Np:
        m += 1
    T_prev = m * (m + 1) // 2
    R = Np - T_prev
    f = min(R / (m + 1), 1.0)

    # analytic density grid
    #x0 = np.loadtxt(data_files[0])[:, 0]
    #xs = np.linspace(x0.min(), x0.max(), 1000)
    xs = np.linspace(-5, 5, 1000)
    analytic = np.zeros_like(xs)
    for n in range(m):
        analytic += shell_cut_density(n, xs)
    if f > 0:
        analytic += f * shell_cut_density(m, xs)

    # color and dash order (lighten green)
    colors     = ['black', '#90ee90', 'cyan', 'orange', 'blue', 'red']
    linestyles = ['-', '--', (0, (5,0.5)), (0, (3.5,0.5)), (0, (2,0.5)), (0, (1,0.5))]

    # 2:1 aspect ratio figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # analytic curve: black solid
    ax.plot(xs, analytic,
            color=colors[0],
            linestyle=linestyles[0],
            lw=4,
            label=f'Analytic (N={Np})')

    # data files with distinct colors & dash styles
    for idx, fname in enumerate(data_files):
        style_idx = idx + 1  # shift past analytic
        data = np.loadtxt(fname)
        # ax.plot(data[:, 0], data[:, 1],
        #         color=colors[style_idx],
        #         linestyle=linestyles[style_idx],
        #         lw=3-0.5*idx,
        #         label=fname)
        xdata = data[:, 0]
        ydata = data[:, 1]
        mask = (xdata >= -5) & (xdata <= 5)
        ax.plot(xdata[mask], ydata[mask],
                color=colors[style_idx],
                linestyle=linestyles[style_idx],
                lw=4 - 0.5 * idx,
                label=fname)

    ax.set_xlabel('$x$ (HO units)')
    ax.set_ylabel(r'$\rho(x,y=0)$')

    # adjust bottom space based on number of files
    bottom_space = 0.25 + 0.02 * len(data_files)
    plt.subplots_adjust(bottom=bottom_space)

    # legend below, one line per entry
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=1,
              frameon=False,
              handlelength=3.5)

    # save and show
    fig.savefig('mpScript_HO.pdf')
    fig.canvas.mpl_connect('close_event', lambda event: sys.exit(0))
    plt.show()

if __name__ == '__main__':
    main()
