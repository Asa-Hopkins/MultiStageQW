import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import scipy
import pickle
import seaborn as sns

sns.set_palette("bright")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 8
plt.rcParams.update({'errorbar.capsize': 1.5, 'lines.markeredgewidth': 0.5})

def bootstrap(a, f, n=100):
    #Estimate mean and std of f(a) via bootstrap sampling.
    samples = [f(a[np.random.randint(0, a.size, a.size)]) for _ in range(n)]
    return np.mean(samples), np.std(samples, ddof=1)

def load_results(file_pattern):
    #Load output files into a nested dict: data[m][n] = float32 array.
    data = {}
    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)
        try:
            _, n, m = filename.split('_')[:3]
            n, m = int(n), int(m)
        except ValueError:
            print(f"Skipping malformed filename: {filename}")
            continue

        if m == 50:
            continue
        
        if n == 18:
            continue

        arr = np.fromfile(filepath, dtype=np.float32)
        data.setdefault(m, {})[n] = arr

    return data

def plot_results(results, outfile, marker = None):
    #Plot log2 median success probability vs n, return regression points.
    reg_points = []
    func = lambda a: np.log2(np.median(a))

    for m in sorted(results):
        xye = np.array([(n, *bootstrap(results[m][n], func)) for n in results[m]]).T
        x, y, yerr = xye[:, np.argsort(xye[0])]
        plt.errorbar(x, y, yerr=yerr, label=f'{m} stage quantum walk', lw=0.75, marker=marker)
        
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        
        reg = scipy.stats.linregress(x, y)
        reg_points.append([m, reg.slope, reg.stderr, m, reg.intercept, reg.intercept_stderr])
    
    plt.xlabel("Number of Spins")
    plt.ylabel(r"$\log_2$ of Success Probability")
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return np.array(reg_points)

def plot_regression(reg_points_list, labels, outfile_scaling, outfile_intercept):
    #Plot scaling exponents and intercepts from regression points.
    for reg_points, label in zip(reg_points_list, labels):
        x, y, yerr = reg_points.T[:3]
        plt.errorbar(x, y, yerr=yerr, label=label, lw=0.75)
    
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.xlabel("Number of Stages")
    plt.ylabel("Scaling Exponent")
    plt.savefig(outfile_scaling, bbox_inches='tight')
    plt.show()

    for reg_points, label in zip(reg_points_list, labels):
        x, y, yerr = reg_points.T[3:]
        plt.errorbar(x, y, yerr=yerr, label=label, lw=0.75)

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        
    plt.xlabel("Number of Stages")
    plt.ylabel("Scaling Intercept")
    plt.savefig(outfile_intercept, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    adam_results = load_results("../results/Adam/output_*_*")
    tim_results  = load_results("../results/Tim/output_*_*")
    inf_results = load_results("../results/Inf/output_*_*")

    reg1 = plot_results(adam_results, 'short_median.pdf')
    reg2 = plot_results(tim_results,  'hard_median.pdf')
    reg3 = plot_results(inf_results,  'inf_median.pdf', marker = 'x')

    labels = ['Short time average', 'Short time average, hard problems', 'Infinite time average']
    plot_regression([reg1, reg2, reg3], labels, 'scaling_cropped.pdf', 'intercept.pdf')
