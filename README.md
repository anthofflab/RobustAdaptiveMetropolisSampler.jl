# RobustAdaptiveMetropolisSampler

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/anthofflab/RobustAdaptiveMetropolisSampler.jl.svg?branch=master)](https://travis-ci.org/anthofflab/RobustAdaptiveMetropolisSampler.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/66r0lf8kcim6fm0o/branch/master?svg=true)](https://ci.appveyor.com/project/anthofflab/robustadaptivemetropolissampler-jl/branch/master)
[![codecov](https://codecov.io/gh/anthofflab/RobustAdaptiveMetropolisSampler.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/anthofflab/RobustAdaptiveMetropolisSampler.jl)

## Overview

This package implements the robust adaptive metropolis (RAM) sampler described in [Vihola (2012)](https://doi.org/10.1007/s11222-011-9269-5) for the [Julia language](https://www.julialang.org).

## Usage

The `RAM_sample` function runs a MCMC sampler on a given log target function. The arguments for the functions are as follows:

```julia
RAM_sample(logtarget, x0, M0, n; opt_α=0.234, γ=2/3, q=Normal(), show_progress=true)
```

* `logtarget` this must be a callable that accepts one parameter which is a vector of values to evaluate the log target function on. The function passed must return the log value of the target function.
* `x0` is a vector of initial values at which the sampler will start the MCMC algorithm. The length of the vector controls the dimensionality of the problem.
* `M0` is the initial co-variance matrix that the sampler should use to scale the new proposal. `M0` can be passed in many different ways:
1) a scalar: an isotropic covariance matrix with diagonal elements `abs2(M0)`.
2) an `AbstractVector`: a diagonal covariance matrix with diagonal elements `abs2.(M0)`.
3) an `AbstractMatrix` (or a `Diagnoal` or an `AbstractPDMat`): a value of any of these types will be interpreted directly as the covariance matrix.
* `n`: the number of elements to be sampled, i.e. the length of the chain.
* `opt_α`: the target acceptance rate the algorithm is trying to hit.
* `γ`: a parameter for the computation of the step size sequence.
* `q`: the proposal distribution.
* `show_progress`: a flag that controls whether a progress bar is shown.

The function returns a `NamedTuple` with three elements:
* `chain`: a `Matrix` with the result chain. Each row is one sample, the columns correspond to the dimensions of the problem.
* `acceptance_rate`: the acceptance rate for the overall chain.
* `M`: the last co-variance matrix used in the algorithm.

A simple example of using the function is

```julia
using Distributions, RobustAdaptiveMetropolisSampler

chain, accrate, S = RAM_sample(
    p -> logpdf(Normal(3., 2), p[1]), # log target function
    [0.],                             # Initial values
    0.5,                              # Scaling factor
    100_000                           # Number of runs
)
```
