using RobustAdaptiveMetropolisSampler, Distributions, LinearAlgebra, VegaLite, DataFrames

chain, accrate, S = RAM_sample(
    p -> logpdf(Normal(3., 2), p[1]), # log target function
    [0.],                             # Initial values
    0.5,                              # Scaling factor
    100_000                           # Number of runs
)

df = DataFrame(p1 = chain[:,1])

df |> @vlplot(:bar, x={:p1, bin=true}, y="count()")
