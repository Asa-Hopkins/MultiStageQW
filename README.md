# Multistage-QW

## About The Project
This repository contains the simulation code for the paper ["Multi-stage quantum walks for finding Ising ground states"](https://arxiv.org/abs/2511.01312).

The main innovations here are an improvement to the classic method of Chebyshev expansion for evaluating matrix exponentials, and a matrix-free method of calculating H|ψ>. The implementation is vectorised and tries to be efficient with cache usage and memory accesses, as memory bandwidth is the bottleneck.

I have now also implemented quantum annealing, using commutator-free exponential time-integration to break a quantum anneal up into quantum walk stages. Parallelism is achieved by simply launching the process multiple times, as separate problem instances are embarrassingly parallel.

The aim of the method here is to provide reasonable choices for hopping rate and evolution time for each stage without needing to tune or optimise them. I invite you to try different heuristics and see what works, but for now only the heuristics in my paper above are available.

## Requirements

- [Eigen 3.4+](https://libeigen.gitlab.io/docs/)
- [Vector Class Library 2](https://github.com/vectorclass/version2)
- The submodule [ApproxTools](https://github.com/Asa-Hopkins/ApproxTools) (see its repo for optional dependencies)

## Getting Started

Clone with submodules and build:

```bash
git clone --recurse-submodules https://github.com/Asa-Hopkins/MultiStageQW/
cd MultiStageQW
g++ -O3 -march=native MultiQW.cpp -o MultiQW
```

To enable verbose output (see [Verbose Mode](#verbose-mode) below), add `-DVERBOSE` to the build command:

```bash
g++ -O3 -march=native -DVERBOSE MultiQW.cpp -o MultiQW
```

## Usage
`./MultiQW n m filename [start] [problems] [output_dir]`

| Argument | Description |
|---|---|
| `n` | Number of spins per problem |
| `m` | Number of walk stages |
| `filename` | Path to the file containing problem instances. Each instance has `n(n+1)/2` entries in double precision: the upper triangle of J followed by the h vector |
| `start` | *(optional)* Index of the first problem to solve (default: 0) |
| `problems` | *(optional)* Number of problems to solve (default: 2000) |
| `output_dir` | *(optional)* Directory to write results to (default: `./results`) |

The `start` and `problems` arguments make it easy to parallelise by running multiple instances over different ranges. `run.sh` handles this automatically — pass it the number of threads and dataset name (`Tim` or `Adam`) to reproduce the results from the paper.

As an explicit example:

```bash
./MultiQW 10 2 data/Adam/SK_10n
```

reproduces one of the data points in Figure 4 of the paper.

Results are written as a binary file of floats, one per problem, giving the success probability for each instance.

## Verbose Mode

Building with `-DVERBOSE` prints the heuristic gamma and t values for each stage, followed by the success probability, for every problem. This is needed for `compare.sh` which generates the file `walks.txt`.

## To-Do
I want to add python bindings at some point for easier integration with other quantum software packages
I also want to try specialising to more types of graph Hamiltonian, I think multiple X-gates or Y-gates should be possible in a similar matrix-free way.
There are some planned improvements to ApproxTools that will affect this repo too, the main improvement would be allowing for double precision simulation.
Finally, I want to try implementing the [commutator-free exponential-time integrator](https://arxiv.org/abs/1102.5071) method for splitting time-dependent Hamiltonians into time-independent parts and see how it compares to ODE methods for performance.

## Contributing
I am open to contributions, discussions, criticism and feature requests. If you are doing work with quantum walks then I'm more than happy to help adapt my code to your use-case.

## References
This work is based on the paper "Finding spin glass ground states using quantum walks" by Adam Callison (https://doi.org/10.1088/1367-2630/ab5ca2), but extends it to allow more than 2 stages.

It uses the [dataset](https://doi.org/10.15128/r21544bp097) mentioned in that paper, which contains 10k spin glass instances to allow for reproducibility. I only use 2000 since the error bars are small enough.

The other dataset available is available [here](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.012611) and was provided to me by Tim Bode. It contains spin glass problems which have been post-selected for having a very small minimum gap, and are arguable the more important instances to check performance on.

For the full list of references, please check the preprint and also the ApproxTools repo.
