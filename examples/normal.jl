using RobustAdaptiveMetropolisSampler, Distributions, LinearAlgebra, VegaLite, DataFrames

d = Normal(3., 2)

function logp(p)
    model1 = d

    return logpdf(model1, p[1])
end

chain, accrate, S = RAM_sample(logp, [0., 0.1], 0.5, 100_000, q = TDist(1))

df = DataFrame(p1 = chain[:,1])

df |> @vlplot(:bar, x={:p1, bin=true}, y="count()")
