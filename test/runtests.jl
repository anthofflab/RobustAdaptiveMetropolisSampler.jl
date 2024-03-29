using RobustAdaptiveMetropolisSampler
using Distributions
using Statistics
using LinearAlgebra
using Random
using Test

@testset "RobustAdaptiveMetropolisSampler" begin

    Random.seed!(10)

    result = RAM_sample(p -> logpdf(Normal(3., 2.), p[1]), [0.], 0.5, 100_000, show_progress=false, output_log_probability_x=true)

    @test length(result.chain) == 100_000
    @test result.acceptance_rate ≈ 0.234 atol = 0.01
    @test mean(result.chain[:,1]) ≈ 3. atol = 0.1
    @test std(result.chain[:,1]) ≈ 2. atol = 0.01
    @test result.log_probabilities_x[1] == logpdf(Normal(3., 2.), result.chain[1])
    @test result.log_probabilities_x[length(result.chain)] == logpdf(Normal(3., 2.), result.chain[length(result.chain)])

    result = RAM_sample(p -> logpdf(Normal(3., 2.), p[1]), [0.], [0.5], 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

    result = RAM_sample(p -> logpdf(Normal(3., 2.), p[1]), [0.], Diagonal([0.5]), 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

    result = RAM_sample(p -> logpdf(Normal(3., 2.), p[1]), [0.], fill(0.5, 1, 1), 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

end
