# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "taichi",
#   "numpy",
#   "pandas",
# ]
# ///

#Throughout, complex arrays are stored with a contiguous real part followed by a contiguous imaginary part 

import taichi as ti
import numpy as np
import pandas as pd
import time

#Return the bessel functions of the first kind, J_m(a) for m up to n
#Needed for Chebyshev series of exp(ix)
def bessel(a, n):
    #Too many terms can cause an overflow, so we rescale if this happens
    max_value = 1
    terms = np.zeros(n+10)
    terms[-2] = 1
    for i in range(n+8, 0, -1):
        terms[i-1] = 2*i / a * terms[i] - terms[i+1]
        if abs(terms[i-1]) > max_value:
            max_value = abs(terms[i-1])
        if max_value > 1e100:
            terms /= 1e100
            max_value /= 1e100

    #Normalise using the fact that the terms sum to 1
    terms[0] /= 2
    norm = 2.0 * np.sum(terms[::2])

    return terms[:n] / norm

def trunc(coeffs, error):
    i = -1
    s = 0
    while s < error:
        s += abs(coeffs[i])
        i -= 1
    i += 2
    return coeffs[:i]

#Generate the Chebyshev coefficients for cos(x) + sin(x)
def sincos(scale, tol = 1e-8):
    coeffs = [1]
    num_terms = int(2*scale + 20)
    while abs(coeffs[-1]) > tol:
        coeffs = bessel(scale, num_terms) * 2
        coeffs[2::4] *= -1;  coeffs[3::4] *= -1
        num_terms *= 2
    return trunc(coeffs, tol)

def unpack_lower_triangle(J_flat, n):
    """
    Expands lower triangle of J to full matrix

    J_flat: shape (n*(n-1)//2, batch)
    Returns J_full: shape (n, n, batch), symmetric, zero diagonal
    """
    batch = J_flat.shape[1]
    J_full = np.zeros((n, n, batch), dtype=np.float32)

    rows, cols = np.tril_indices(n, k=-1)

    J_full[rows, cols, :] = J_flat
    J_full[cols, rows, :] = J_flat

    return J_full/2

ti.init(arch=ti.cpu)  # use ti.gpu if available

@ti.func
def get_spin(state_idx: ti.i32, bit: ti.i32) -> ti.f32:
    b = (state_idx >> bit) & 1
    return 2.0 * b - 1.0

@ti.kernel
def build_hamiltonian_batch(
    J: ti.types.ndarray(),      # shape (n, n, batch)
    h: ti.types.ndarray(),      # shape (n, batch)
    H_out: ti.types.ndarray(),  # shape (batch, N) where N = 2^n
    delta2_out: ti.types.ndarray(), # shape (batch,)    n: ti.i32,
    n: ti.i32,
    N: ti.i32,
):
    #Calculates H_P from the Ising parameters for each problem in the batch
    #I use the slow O(Nn^2) method as it parallelises better

    batch = J.shape[2]
    
    # Calculate average energy difference for each Hamiltonian.
    for b in range(batch):
        delta2 = 0.0
        for i in range(n):
            delta2 += 4.0 * h[i, b] * h[i, b]
            for j in range(n):
                delta2 += 16.0 * J[i, j, b] * J[i, j, b]
        delta2_out[b] = delta2

    #Calculate the energy levels for each Hamiltonian
    for b, state in ti.ndrange(batch, N):
        energy = 0.0
        for i in range(n):
            s_i = get_spin(state, i)
            for j in range(n):
                s_j = get_spin(state, j)
                energy += s_i * J[i, j, b] * s_j
            energy += s_i * h[i, b]
        H_out[b, state] = energy

@ti.kernel
def Clenshaw_step(b1: ti.types.ndarray(), b2: ti.types.ndarray(),
                  psi: ti.types.ndarray(), H_P: ti.types.ndarray(),
                  n: ti.i32, scales: ti.types.ndarray(), gamma: ti.types.ndarray(), coef: ti.f32):
    #Sets b2 = coef*psi + 2*(H @ b1) - b2
    #where H = scale*(H_P - gamma*H_G)
    for i in range(b1.shape[0]):
        s = 0.0
        for k in range(n):
            s += b1[i ^ (1 << k)]
        b2[i] = coef*psi[i] + 2*(H_P[i]*b1[i] - gamma[i>>n]*s)*scales[i>>n] - b2[i]

def Clenshaw(coeffs: ti.types.ndarray(), psi: ti.types.ndarray(),
             H_P: ti.types.ndarray(), gamma: ti.types.ndarray(), scales: ti.types.ndarray(), n: ti.i32):
    #Calculates exp(iH) @ psi, where "coeffs" gives the coefficients of exp(ix) in the Chebyshev basis
    N = psi.shape[0] // 2
    b1 = np.zeros(2*N, dtype=np.float32)
    b2 = np.zeros(2*N, dtype=np.float32)
    for r in range(len(coeffs)-1,0,-1):
        if r&1:
            Clenshaw_step(b1[:N], b2[:N], psi[N:], H_P, n, scales, gamma, -coeffs[r])
            Clenshaw_step(b1[N:], b2[N:], psi[:N], H_P, n, scales, gamma, coeffs[r])
        else:
            Clenshaw_step(b1[:N], b2[:N], psi[:N], H_P, n, scales, gamma, coeffs[r])
            Clenshaw_step(b1[N:], b2[N:], psi[N:], H_P, n, scales, gamma, coeffs[r])
        b2, b1 = b1, b2
    
    Clenshaw_step(b1[:N], b2[:N], psi[:N], H_P, n, scales/2, gamma, coeffs[0])

    Clenshaw_step(b1[N:], b2[N:], psi[N:], H_P, n, scales/2, gamma, coeffs[0])
    return b2

#Uses an estimate of the energy spread to calculate gammas
def compute_gammas(n, m, E_est):
    gammas = np.zeros((m, E_est.size), np.float32)
    for i in range(0,m):
        gammas[i] = E_est / np.tan(np.pi * (i+1) / (2*m + 2)) / 2 / n
    return gammas


#Use a greedy local search from random samples to estimate energy spread
@ti.kernel
def greedy_est(
    H_P: ti.types.ndarray(), # (batch, N)
    E_est: ti.types.ndarray(), # (batch)
    samples: ti.i32,
    n: ti.i32
):
    batch = H_P.shape[0]
    N = 2**n

    # Each Hamiltonian is processed independently in parallel
    for b in range(batch):
        min_energy = ti.math.inf
        max_energy = -ti.math.inf

        for _ in range(samples):
            # ti.random() is uniform in [0, 1)
            state = ti.cast(ti.random(ti.f32) * N, ti.i32)
            energy = H_P[b, state]

            state2 = state
            energy2 = energy

            #Local search for minimum
            for __ in range(n):
                # Search all single-spin-flip neighbours
                for i in range(n):
                    neighbour = state ^ (1 << i)
                    neighbour_energy = H_P[b, neighbour]

                    if neighbour_energy < energy:
                        state = neighbour
                        energy = neighbour_energy

            
            #Local search for maximum
            for __ in range(n):
                # Search all single-spin-flip neighbours
                for i in range(n):
                    neighbour = state2 ^ (1 << i)
                    neighbour_energy = H_P[b, neighbour]

                    if neighbour_energy > energy2:
                        state2 = neighbour
                        energy2 = neighbour_energy

            min_energy = ti.min(min_energy, energy)
            max_energy = ti.max(max_energy, energy2)

        E_est[b] = max_energy - min_energy

#Number of qubits
n = 15
m = 5
N = 2**n

#Ising model parameters
total_params = n*(n+1)//2
J_params = n*(n-1)//2
h_params = n

#Load all instances at once
instances = np.fromfile(f"./data/Adam/SK_{n}n").reshape((-1, total_params))
batch_size = instances.shape[0]

J = instances[:, :J_params]

#Easiest to unpack to full matrix form for now
J_full = unpack_lower_triangle(np.ascontiguousarray(J.T, dtype=np.float32), n)

#Taichi doesn't like h for some reason so I need to make it contiguous
h = np.ascontiguousarray(instances[:, J_params:].T, dtype=np.float32)

#Build problem Hamiltonians, and energy spreads
H_P = np.zeros((batch_size, N), dtype=np.float32)
delta2 = np.zeros(batch_size, dtype=np.float32)
build_hamiltonian_batch(J_full, h, H_P, delta2, n, N)
#Can calculate from J
test = 0

#We have to scale each Hamiltonian separately. We first need to find the largest spectral norm that appears
sizes = np.max(np.abs(H_P), axis = 1)

exact_spreads = np.max(H_P, axis = 1) - np.min(H_P, axis = 1)

E_est = np.zeros_like(exact_spreads)

#Use 10 random samples for now
greedy_est(H_P, E_est, 10, n)

gammas = compute_gammas(n, m, E_est)

#Also need ground states later
E_min = np.argmin(H_P, axis = 1)

#We just treat H_P as a huge vector since it's diagonal
H_P = H_P.flatten()

#Start in uniform superposition
psi = np.ones(2*N*batch_size, dtype=np.float32)/(N**0.5)
psi[N*batch_size:] = 0

t = time.time()
for i in range(m):
    gamma = gammas[i]

    last_denom = gammas[i] * 2 * E_est

    gamma_last = 1 if i == 0 else gammas[i - 1] / np.sqrt(1 + gammas[i - 1]**2)
    gamma_next = 0 if i == (m - 1) else gammas[i + 1] / np.sqrt(1 + gammas[i + 1]**2)
    dE = 2*n*(gamma_last - gamma_next)

    first_t = np.sqrt(2*dE/delta2)
    last_t = np.sqrt(dE/last_denom)

    times = first_t if i == 0 else np.max((first_t, last_t), axis = 0)
    times *= 1.5
    
    #This is the largest spectral norm
    max_bound = np.max(times*(sizes + n*gamma))
 
    #Our chebyshev polynomial is exp(i*max_bound*x), so every input gets scaled by max_bound
    #Suppose one problem is bounded by "bound", and we want to evolve for time t
    #So we need to pass (H*t)/max_bound in to our polynomial to get exp(itH)
    scales = times / max_bound

    #Apply the Clenshaw algorithm
    psi = Clenshaw(sincos(max_bound), psi, H_P, gamma, scales, n)
success_probs = []

#Read out success probabilities
for i in range(batch_size):
    min_loc = E_min[i]
    success_probs.append(np.sum(psi[N*i+min_loc]**2 + psi[N*(i+batch_size)+min_loc]**2))
print(time.time() - t)
print(np.log2(np.median(success_probs)))
