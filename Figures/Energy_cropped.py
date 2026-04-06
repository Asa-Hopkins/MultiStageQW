import pandas as pd
import itertools
import time
import pickle
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette("bright")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14

import numpy as np
import scipy

def grid(n):
    # Construct adjacency matrix for a hypercube graph of dimension n.
    # The 2^i'th off-diagonals are nonzero, with a repeating pattern of
    # j ones followed by j zeros for the j'th diagonal.
    N = 2**n
    diags, offsets = [], []
    for i in range(n):
        diags.append(-(np.indices([N - i - 1], dtype=np.int32) + 2**i & 2**i) // 2**i)
        offsets.append(2**i)
    A = scipy.sparse.diags(diags, offsets=offsets, format='csr', shape=(N, N), dtype=np.float32)
    return A + A.T

def all_states(n):
    # Returns a (2^n, n) array where each row is a spin configuration of ±1.
    lists = [[-1, 1]] * n
    return np.fromiter(
        itertools.chain.from_iterable(itertools.product(*lists)), int
    ).reshape(-1, n)

def P(t, diag, eigs, Psi0, e0, H):
    # Fixed time, single stage: returns success probability and energy at each t.
    temp = eigs @ Psi0
    diags = np.exp(-1j * t[:, None] * diag[None, :])
    temp = (diags * temp[None, :]).T
    psit = eigs.T @ temp
    res = np.abs(psit[e0])**2
    E = np.sum(psit.conj() * (H @ psit), axis=0)
    return res, np.real(E)

def heur(n):
    # Heuristic hopping rate for n qubits, from the Callison paper.
    return 0.887 * (2**0.5 * (n * (n + 3))**0.5) * scipy.special.erfinv(1 - 1/2**n) / 2 / n

def SpinGlass2(n, m, plot_energy=True, write=True, num=2000):
    N = 2**n
    H_G = grid(n)
    Psi0 = 1 / np.sqrt(N) * np.ones(N)

    entries = np.array(pd.read_csv('../qwspinglass_data/sk_instances.csv'))

    probs = []
    t = time.time()

    for numero in np.arange((n - 5) * 10000, (n - 5) * 10000 + num):
        J = np.load(f"../qwspinglass_data/sk_instances/{entries[numero, 0]}.Jmat.npy")
        h = -np.load(f"../qwspinglass_data/sk_instances/{entries[numero, 0]}.hvec.npy")
        J = -np.tril(J + J.T) / 2

        states = all_states(n).T
        H_P = np.sum(states * (J @ states), axis=0) + np.sum(states * h[:, None], axis=0)

        arg = np.argmin(H_P)
        H_P = scipy.sparse.diags(H_P, format='csr')

        gammas = heur(n) / np.tan(np.arange(1, m + 1) * np.pi / (2 * (m + 1)))
        Psit = np.copy(Psi0)

        dE = 4 * np.sum(J * J) + 2 * np.sum(h * h)
        point = np.sqrt(2 * n / dE)
        print(dE, np.mean(H_P.data**2), dE / np.mean(H_P.data**2), point)

        E_G = [-n]
        times = []
        gammas = np.ones(5) * gammas[0]

        for gamma in gammas:
            H = gamma * H_G + H_P
            vals, eigs = np.linalg.eigh(H.toarray())
            point2 = np.sqrt(2 * n / dE)

            if times:
                times.extend(times[-1] + np.linspace(0, point2, 100))
            else:
                times.extend(np.linspace(0, point2, 100))

            _, b = P(np.linspace(0, point2, 100), vals, eigs.T, Psit, arg, H_P)
            E_G.extend(E_G[-1] + (b[0] - b) / gamma)
            plt.axvline(times[-1] / point, linestyle='dashed', alpha=0.6, lw=0.75)

            Psit = eigs @ (np.exp(-1j * vals * point2) * (eigs.T @ Psit))

        times = np.array(times)
        temp = np.linspace(0, point, 100)

        plt.plot(times / point, E_G[1:], label=r"Energy of $\hat{H}_G$ observable", lw=0.75)
        plt.plot(temp / point, -n + dE * temp**2, label=r"$2^{nd}$ Order Approximation", lw=0.75)
        plt.xlabel(r"Time (normalised by $t_s$)")
        plt.ylabel("Energy")
        plt.savefig('Energy_cropped.pdf', bbox_inches='tight')
        plt.show()
        return


SpinGlass2(10, 1)
