module RobustAdaptiveMetropolisSampler

using LinearAlgebra, Random, Distributions, PDMats, ProgressMeter

export RAM_sample

# The following methods cover different ways to pass in a co-variance matrix
function RAM_sample(logtarget, x0::AbstractVector{<:Number}, M0::AbstractMatrix{<:Real}, n::Int; kwargs...)
    return RAM_sample(logtarget, x0, PDMat(M0), n; kwargs...)
end

function RAM_sample(logtarget, x0::AbstractVector{<:Number}, M0::AbstractVector{<:Real}, n::Int; kwargs...)
    return RAM_sample(logtarget, x0, PDiagMat(abs2.(M0)), n; kwargs...)
end

function RAM_sample(logtarget, x0::AbstractVector{<:Number}, M0::Diagonal{<:Real}, n::Int; kwargs...)
    return RAM_sample(logtarget, x0, PDiagMat(diag(M0)), n; kwargs...)
end

function RAM_sample(logtarget, x0::AbstractVector{<:Number}, M0::Real, n::Int; kwargs...)
    return RAM_sample(logtarget, x0, ScalMat(length(x0), abs2(M0)), n; kwargs...)
end

# Actual sampling code

function RAM_sample(logtarget, x0::AbstractVector{<:Number}, M0::AbstractPDMat, n::Int; opt_α=0.234, γ=2/3, q=Normal(), show_progress::Bool=true)
    length(x0) == size(M0, 1) || error("Covariance matrix M0 must match size of x0.")
    n > 0 || error("n must be larger than 0.")
    0 < opt_α < 1 || error("opt_α must be between 0 and 1.")
    0.5 < γ <= 1 || error("γ must be between 0.5 and 1.")

    x = copy(x0)
    s = cholesky(M0)

    d = length(x0)

    u = zeros(d)
    y = zeros(d)

    log_probability_x = logtarget(x0)

    output_chain = Matrix{Float64}(undef, n, d)

    stats_accepted_values = 0

    progress_meter = Progress(n, show_progress ? 0.1 : Inf)

    # This is a pre-allocated vector used in the loop
    scaled_proposal_vector = Vector{Float64}(undef, d)

    for i in 1:n
        # Step R1
        rand!(q, u)
        
        y[:] .= x .+ mul!(scaled_proposal_vector, s.L, u)

        # Step R2
        log_probability_y = logtarget(y)

        α = min(1, exp(log_probability_y - log_probability_x))

        if α > rand()
            stats_accepted_values += 1
            x, y = y, x
            log_probability_x = log_probability_y
        end

        # Step R3

        # This is taken from the second paragraph of section 5
        η = min(1, d * i^-γ)

        # Compute the new covariance matrix
        M = s.L * (I + η * (α-opt_α) * (u * u') / norm(u)^2 ) * s.L'

        # The paper has a proof that M is symmetric, so we declare that fact
        # to work around numerical rounding errors
        s = cholesky(Symmetric(M))

        output_chain[i, :] .= x

        next!(progress_meter; showvalues = [(:acceptance_rate,stats_accepted_values/i)])
    end

    return (chain=output_chain, acceptance_rate=stats_accepted_values/n, M=s.L * s.L')
end

end
