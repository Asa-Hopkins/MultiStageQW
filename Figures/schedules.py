import numpy as np
from matplotlib import pyplot as plt
import scipy
import csv
from matplotlib.ticker import MaxNLocator, MultipleLocator

import seaborn as sns
sns.set_palette("bright")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14
plt.rcParams.update({'errorbar.capsize': 1.5})
plt.rcParams.update({'lines.markeredgewidth': 0.5})

def lookups():
    #We need functions that let us map a ratio of H_P and H_G to a point in the annealing schedule
    #For reference, we will use D-Wave's Fast Annealing schedule on Advantage 7.1
    s = []
    A = []
    B = []
    with open('../data/DWave/Advantage2_1_6.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            x,y,z,_ = lines
            if x == 's':
                continue
            s.append(float(x))
            A.append(float(y))
            B.append(float(z))
    if 0 in A:
        stop = A.index(0) + 1
    else:
        stop = len(A)
    s = np.array(s[:stop])
    A = np.array(A[:stop])
    B = np.array(B[:stop])

    #Given a ratio, we want the value of A and B at that point in the schedule.
    #We make one function that maps the ratio (gamma) to s, then map s to A and B
    gamma = A/B
    return s, gamma


x = [0,1]
y = [1.22758,1.22758]
plt.plot(x, y, label = f'Short time average', lw = 0.75)

plt.xlabel("Anneal Progress")
plt.ylabel("$\gamma$")
plt.ylim((0,5.33))
plt.savefig('Walk1.pdf', bbox_inches='tight')
plt.show()

params = [0.187075, 4.58138, 0.272453, 2.12623, 0.449822, 1.22758, 0.704668, 0.708742]
times = params[::2]
gammas = params[1::2]

x = np.cumsum([0] + times)
x/=x[-1]
x = np.repeat(x,2)
x = x[1:-1]
y = np.repeat(gammas,2)

plt.plot(x, y, label = f'Short time average', lw = 0.75)

plt.xlabel("Anneal Progress")
plt.ylabel("$\gamma$")
plt.ylim((0,5.33))
plt.savefig('Walk5.pdf', bbox_inches='tight')
plt.show()

x, y = lookups()

plt.plot(x, y, label = f'Short time average', lw = 0.75)

plt.xlabel("Anneal Progress")
plt.ylabel("$\gamma$")
plt.savefig('WalkInf.pdf', bbox_inches='tight')
plt.show()
