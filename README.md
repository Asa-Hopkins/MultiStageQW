# Multistage-QW

## About The Project
This repository contains the code used for simulating multi-stage quantum walks in my paper, ["Multi-stage quantum walks for finding Ising ground states"](https://arxiv.org/abs/2511.01312).
The main innovations here are an improvement to the classic method of Chebyshev expansion for evaluating matrix exponentials, and a matrix-free method of calculating H|ψ>. The implementation is vectorised and tries to be efficient with cache usage and memory accesses, as memory bandwidth is the bottleneck.

The aim of the method here is to provide reasonable choices for hopping rate and evolution time for each stage without needing to tune or optimise them. I invite you to try different heuristics and see what works, but for now only the heuristics in my paper above are available.

## Requirements
- [Eigen 3.4+](https://libeigen.gitlab.io/docs/)
- [Vector Class Libary 2](https://github.com/vectorclass/version2)
- The submodule [ApproxTools](https://github.com/Asa-Hopkins/ApproxTools) has some optional requirements 

## Getting Started
Once Eigen and VCL2 are installed, then clone everything with
`git clone --recurse-submodules https://github.com/Asa-Hopkins/MultiStageQW/
cd MultiStageQW`

Then build with 
`g++ -O3 -march=native MultiQW.cpp -o MultiQW`

## Example Usage
The inputs to the program are:
`n` - number of spins per problem
`m` - number of walk stages
`filename` - the name of the file containing problem instances. These instances should have `n(n+1)/2` entries each, in double precision. The first `(n-1)*n` are the upper triangle of the J matrix, and the rest are the h vector.

The next two inputs are optional:
`start` - which problem to start one
`problems` - how many problems to solve

These are provided for easier multi-threading, since doing multiple problems at once is easily parallelised. The `run.sh` file allows for easier multi-threading. It requires some modification to use as it loops through a given list of `n` and `m` values for a given file.

As an explicit example, the command
`./MultiQW 10 2 data/AdamData/SK_10n`
will recreate the data for one of the data points in Figure 4 of my paper.
Currently the code spits out a unique file for each (n,m) pair, which is a bit messy but works. These files contain a floating point value for each problem, describing the probability of finding the ground state for that problem

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
