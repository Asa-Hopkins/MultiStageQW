import numpy as np
import scipy
import itertools
from matplotlib import pyplot as plt
import csv
import re
import qutip as qt
import seaborn as sns

sns.set_palette("bright")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14
plt.rcParams.update({'errorbar.capsize': 1.5, 'lines.markeredgewidth': 0.5})

def lookups(cheb = False):
    #We need functions that let us map a ratio of H_P and H_G to a point in the annealing schedule
    #For reference, we will use D-Wave's Fast Annealing schedule on Advantage2 1.6
    #Multiply by 2*pi to write in terms of h-bar rather than h
    s, A, B = [], [], []
    with open('../data/DWave/Advantage2_1_6.csv') as f:
        for row in csv.reader(f):
            if row[0] == 's':
                continue
            s.append(float(row[0]))
            A.append(float(row[1])*np.pi)
            B.append(float(row[2])*np.pi)

    # Trim to where A reaches zero
    if 0 in A:
        stop = A.index(0) + 1
    else:
        stop = len(A)
    s, A, B = [np.array(x[:stop]) for x in (s, A, B)]
    if cheb:
        V = np.polynomial.chebyshev.chebvander(2*s - 1, 50)
            
        A_cheb, r, *_ = np.linalg.lstsq(V, A, rcond=None)
        
        #Output as C array
        print(f"{{{', '.join(map(str, A_cheb))}}}")

        B_cheb, r, *_ = np.linalg.lstsq(V, B, rcond=None)
        print(f"{{{', '.join(map(str, B_cheb))}}}")
        

    gamma = A / B
    gs = scipy.interpolate.CubicSpline(gamma[::-1], s[::-1])
    sa = scipy.interpolate.CubicSpline(s, A)
    sb = scipy.interpolate.CubicSpline(s, B)
    return gs, sa, sb

gs, sa, sb = lookups(cheb = True)
#die

# Define time-dependent coefficients A(t) and B(t)
# Linear schedule for now

def A(t, args):
    return float(sa(t / args['T']))

def B(t, args):
    return float(sb(t / args['T']))

def parse_results_file(filepath):
    #Parse walks.txt into dict: data[(m, problem_num)] = {times, gammas, success_prob}.
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    header_re = re.compile(r"Running program with .*m=(\d+),.*start_point=(\d+)")
    data = {}
    i = 0

    while i < len(lines):
        match = header_re.match(lines[i])
        if not match:
            i += 1
            continue

        m = int(match.group(1))
        start = int(match.group(2))
        i += 1

        for problem_num in [start, start + 1]:
            times, gammas = [], []
            for _ in range(m):
                t, g = map(float, lines[i].split())
                times.append(t)
                gammas.append(g)
                i += 1
            data[(m, problem_num)] = {
                "times": times,
                "gammas": gammas,
                "success_prob": float(lines[i])
            }
            i += 1

    return data

def grid(n):
    #Construct adjacency matrix for a grid of dimension n
    #It does this by reproducing a pattern that occurs down the diagonals

    #For dimension n, there are n diagonals with nonzero elements
    #These are the 2**i'th off-diagonals for every integer i up to n
    #The pattern is then that for the j'th diagonal, there are j -1's followed by j 0's
    #This repeats until the end of the diagonal
    N = 2**n
    diags = []
    offset = []
    for i in range(0,n):
        diags.append(-(np.indices([N - i - 1], dtype=np.int32) + 2**i & 2**i) // 2**i)
        offset.append(2**i)
    A = scipy.sparse.diags(diags,offsets = offset,format = 'csr', shape = (N, N), dtype = np.float32)
    A += A.T
    return A

def all_states(n):
    #Generates all possible states for a system of n spins.
    #Returns a 2D array of shape (2**n, n), where each row represents a different state.
    lists = []
    for i in range(n):
        lists.append([1,-1])
    return np.fromiter(itertools.chain.from_iterable(itertools.product(*lists)),int).reshape(-1,n)

def anneal_time(T):
    #We want a way to calculate the wall clock annealing time from the arbitrary units used above
    #Output is in seconds
    
    #We'll use a linear anneal schedule
    points = 1000

    #We just use the midpoint rule
    times = T*np.linspace(1/points/2, 1 - 1/points/2,points)
    
    dt = times[1] - times[0]

    #We want to evolve under H = A*H_G + B*H_P for some time dt
    #We calculate the ratio A/B and look up the real A and B values we're using
    #A and B have been scaled to real_A and real_B by multiplying by real_A/A
    #We need to scale up dt by A/real_A, remembering that real_A is in GHz
    
    gs, sa, sb = lookups()
    
    total_time = 0
    args = {'T':T}

    #Linear Anneal
    
    for t in times:
        gamma = A(t, args)/B(t, args)
        s = gs(gamma)
        real_A = sa(s)
    
        total_time += A(t,args) / real_A * 10**-9 * dt


    return total_time

def walk_time(m, problem, walk_data):
    #Wall clock time and success probability for a quantum walk run
    d = walk_data[(m, problem)]

    total = 0
    if m == 1600:
        #Approximate infinite stage anneal schedule by smoothing the 1600 stage schedule
        t = []
        a = []
        b = []
        for dt, g in zip(d['times'], d['gammas']):
            s = gs(g)
            if s < 0:
                s = 0
            total += g / sa(s) * 1e-9 * dt
            t.append(total)
            a.append(sa(s))
            b.append(sb(s))
        t = np.array(t)

        #Fit Chebyshev polynomial
        V = np.polynomial.chebyshev.chebvander(2*t/max(t) - 1, 30)
        A_cheb, r, *_ = np.linalg.lstsq(V, a, rcond=None)
        #Output as C array
        print(f"{{{', '.join(map(str, A_cheb))}}}")
        
        B_cheb, r, *_ = np.linalg.lstsq(V, b, rcond=None)
        print(f"{{{', '.join(map(str, B_cheb))}}}")
    else: 
        for dt, g in zip(d['times'], d['gammas']):
            s = gs(g)
            if s < 0:
                s = 0
#            total += g / sa(s) * 1e-9 * dt
            total += 1e-9 * dt / sb(s)
    # Factor of 1.5 since we sample uniformly from [t_s, 2*t_s]
    return 1.5 * total, d['success_prob']

def run_annealing(n, problem_num, walk_data, m_values):
    #Simulate quantum annealing and compare to walk results for one problem.
    N = 2**n
    H_G = grid(n)
    states = all_states(n).T

    entries = np.fromfile(f"../data/Adam/SK_{n}n")
    stride = (n * (n + 1)) // 2

    # Load the specific problem
    index = problem_num * stride
    J = np.zeros((n, n))
    h = np.zeros(n)
    k = 0
    for i in range(n):
        h[i] = entries[index + n*(n-1)//2 + i]
        for j in range(i):
            J[i, j] = entries[index + k]
            k += 1

    # Normalise J
    J = np.tril(J + J.T)
    #D-Wave only supports J's and h's in this range
    scale = np.max([np.max(np.abs(J)), np.max(np.abs(h)) / 2])
    J /= scale
    h /= scale

    H_P_diag = np.sum(states * (J @ states), axis=0) + np.sum(states * h[:, None], axis=0)
    arg = np.argmin(H_P_diag)

    # Collect walk comparison points
    walk_times, walk_probs = [], []
    for m in m_values:
        if (m, problem_num) in walk_data:
            t, p = walk_time(m, problem_num, walk_data)
            walk_times.append(1e9 * t * scale)
            walk_probs.append(p)
    return
    # Set up QuTiP operators
    H_P_qt = qt.Qobj(scipy.sparse.diags(H_P_diag))
    H_G_qt = qt.Qobj(H_G)
    H = [[H_G_qt, A], [H_P_qt, B]]

    E0 = qt.Qobj(np.eye(N)[[arg]])
    psi0 = qt.Qobj(np.ones(N) / N**0.5)

    anneal_x, anneal_y = [], []
    for T in [2**(a/8) for a in range(-36,32)]:
        args = {'T': T}
        t_list = np.linspace(0, T, 2 + int(T)*2)
        result = qt.sesolve(H, psi0, t_list, args=args)
        prob = qt.expect(E0.dag() * E0, [result.states[-1]])[0]
        anneal_x.append(anneal_time(T) * 1e9)
        anneal_y.append(float(prob))
        print(f"Problem {problem_num}, T={T:.2f}: real time={anneal_time(T):.3e}s, p={prob:.4f}")

    plt.figure()
    plt.plot(anneal_x, anneal_y, label='Quantum Anneal', lw=0.75)
    plt.scatter(walk_times, walk_probs, marker='x', lw=0.5, label='Multi-stage Quantum Walks')
    plt.xlabel('Time (ns)')
    plt.ylabel('Success Probability')
    plt.savefig(f'anneal_{problem_num}.pdf', bbox_inches='tight')
    plt.show()


walk_data = parse_results_file("walks.txt")
m_values = [1, 2, 5, 10, 20, 50, 200, 400, 800, 1600]

for problem in [1780, 1781]:
    run_annealing(8, problem, walk_data, m_values)
