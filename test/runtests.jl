using RobustAdaptiveMetropolisSampler
using Distributions
using Statistics
using Test

@testset "RobustAdaptiveMetropolisSampler" begin

result = RAM_sample(p->logpdf(Normal(3.,2.), p[1]), [0.], 0.5, 100_000, show_progress=false)

@test length(result.chain) == 100_000
@test result.acceptance_rate ≈ 0.234 atol=0.01
@test mean(result.chain[:,1]) ≈ 3. atol=0.1
@test_broken std(result.chain[:,1]) ≈ 2.

end
