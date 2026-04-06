import scipy
from scipy import stats
import numpy as np
import itertools

def var(J):
    return 2*np.sum(J*J)

def skew(J):
    return 8*np.sum(J*(J@J))

def kurt(J):
    comp1 = 48*np.sum((J@J@J)*J)
    comp2 = 12*np.sum(J*J)**2
    comp3 = 32*np.sum(J**4)
    comp4 = -96*np.sum((J*J)@(J*J))
    return comp1 + comp2 + comp3 + comp4

def mom5(J):
    comp1 = 384 * np.sum(J * (J @ J @ J @ J))
    comp2 = 160 * np.sum(J*J) * np.sum((J@J)*J)
    comp3 = -1920 * np.sum(((J @ J) * J) @ (J*J))
    comp4 = 1280 * np.sum((J@J)*J*J*J)
    return comp1 + comp2 + comp3 + comp4

def statistic(x, axis):
    #underlying calculation of the Jarque Bera statistic
    #Taken from the scipy documentation
    s = stats.skew(x, axis=axis)
    k = stats.kurtosis(x, axis=axis)
    return x.shape[axis]/6 * (s**2 + k**2/4)

def all_states(n):
    #Generates all possible states for a system of n spins.
    #Returns a 2D array of shape (2**n, n), where each row represents a different state.
    lists = []
    for i in range(n):
        lists.append([1,-1])
    return np.fromiter(itertools.chain.from_iterable(itertools.product(*lists)),int).reshape(-1,n)

results = []
np.random.seed(745528452849154 & (2**32 - 1))

def p_val(J):
    
    #Perform Jarque-Bera normality test
    #Must be stochastic for N < 2000
    if n <= 10:
        H_P = np.sum(states * (J @ states), axis=0)
        H_P = np.float32(H_P)/var(J)**0.5
        #Verify our calculation of JB is correct
        #print(JB, statistic(H_P, axis = 0))
        
        res = stats.monte_carlo_test(H_P, stats.norm.rvs, statistic, alternative='greater', vectorized=True, n_resamples=1000)
        pvalue = res.pvalue
    else:
        #Calculate test statistic from skew and kurtosis
        v = var(J)
        s = skew(J)/v**(3/2)
        k = kurt(J)/v**2

        JB = 2**n / 6 * (s**2 +(k - 3)**2/4)
        dist = stats.chi2(df=2)
        pvalue = dist.sf(JB)
    return pvalue

for n in [5,8,10,20]:
    if n <= 10:
        states = all_states(n).T
    for problem in range(3):
        results.append([])
        for run in range(1000):
            if run % 10 == 0:
                print(run)
            #Generate random instance
            if problem == 0:
                #These problems have high excess kurtosis at all sizes tested here
                J = np.random.randint(-2,2,(n,n))
            if problem == 1:
                #For larger problems these problems have 0 excess kurtosis
                J = np.random.uniform(-1,1,(n,n))
            if problem == 2:
                #Spin-glass problems should look normally distributed at any size 
                J = np.random.randn(n,n)

            J = J + J.T
            J -= np.diag(np.diag(J))

            results[-1].append(p_val(J))

print(np.array([np.sum(np.array(x) < 0.05) for x in results]).reshape(-1,3))

#The next section is checking how much adding extra moments affects the extreme value
import pymaxent
from matplotlib import pyplot as plt
v = var(J)
s = skew(J)/v**(3/2)
k = kurt(J)/v**2

print(v, s, k)
mu = [1, 0, 1, s, k, mom5(J)/v**(5/2)]

sol, lambdas = pymaxent.reconstruct(mu, bnds = [-10,10])
print(lambdas)
sol2, lambdas = pymaxent.reconstruct(mu[:3], bnds = [-10,10])
print(lambdas)

x = np.linspace(-10,10,1000)
plt.plot(x, sol(x))
plt.plot(x, sol2(x))

#Turn the above functions into actual scipy.stats pdfs
from scipy.stats import rv_continuous

class MyCustomDist(rv_continuous):
    def _pdf(self, x):
        return sol(x)

class MyCustomDist2(rv_continuous):
    def _pdf(self, x):
        return sol2(x)

#Initialize the distribution
#a is the lower bound, b is the upper bound
my_dist = MyCustomDist(a=-10, b=10, name='my_dist')

my_dist2 = MyCustomDist2(a=-10, b=10, name='my_dist')

N = 2**n

loc = my_dist.ppf(1 - 1/N)
scale = 1/(N * sol(loc))

gumbel = scipy.stats.gumbel_r(loc = loc, scale = scale)

plt.plot(x, gumbel.pdf(x))
plt.plot(x, N * my_dist2.cdf(x)**(N - 1) * my_dist2.pdf(x))
plt.plot(x, N * my_dist.cdf(x)**(N - 1) * my_dist.pdf(x))
plt.show()

x = np.linspace(-2,2,1000)
#Finally, see how quickly the true extreme value distribution approaches the Gumbel distribution
gumbel = scipy.stats.gumbel_r()
plt.plot(x, gumbel.pdf(x))
for N in [2**i for i in range(5,26,5)]:
    loc = stats.norm.ppf(1 - 1/N)
    scale = stats.norm.ppf(1 - 1/N/np.e) - loc
#    print(loc, scale)
#    print(1/(N * stats.norm.pdf(loc)))
    print(scipy.stats.entropy(gumbel.pdf(x), N*stats.norm.cdf(x*scale + loc)**(N-1) * stats.norm.pdf(x*scale + loc)))
#    scale = 1/(N * stats.norm.pdf(loc))
    #plt.plot(x, stats.norm.cdf(x*scale + loc)**N)

#plt.show()

