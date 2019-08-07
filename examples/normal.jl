using MCMCRAMSampler, Distributions, LinearAlgebra, VegaLite, DataFrames

d = Normal(3., 2)

function logp(p)
    model1 = d

    return logpdf(model1, p[1])
end

chain = MCMCRAMSampler.sample(logp, [0., 0.1], Diagonal([0.5, 0.5]), 100_000)

df = DataFrame(p1 = chain[10_000:10:end,1])

df |> @vlplot(:bar, x={:p1, bin=true}, y="count()")
