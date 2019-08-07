using RobustAdaptiveMetropolisSampler, Distributions, LinearAlgebra, VegaLite, DataFrames

logp(p) = logpdf(Normal(3., 2), p[1])

chain, accrate, S = RAM_sample(logp, [0.], 0.5, 100_000, q = TDist(1))

df = DataFrame(p1 = chain[:,1])

df |> @vlplot(:bar, x={:p1, bin=true}, y="count()")
