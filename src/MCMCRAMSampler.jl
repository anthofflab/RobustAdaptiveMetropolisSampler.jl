module MCMCRAMSampler

using LinearAlgebra, Random

function sample(logtarget, x0::AbstractVector{<:Number}, s0, n::Int; opt_α=0.234, γ=0.75)
    n > 0 || error("n must be larger than 0.")
    0 < opt_α < 1 || error("opt_α must be between 0 and 1.")
    0.5 < γ <= 1 || error("γ must be between 0.5 and 1.")

    x = copy(x0)
    s = cholesky(s0) # Probably wrong

    d = length(x0)

    u = zeros(d)
    y = zeros(d)

    log_probability_x = logtarget(x0)

    output_chain = Matrix{Float64}(undef, n, d)

    stats_accepted_values = 0

    for i in 1:n
        # Step R1
        randn!(u)
        y[:] .= x .+ s.L * u

        # Step R2
        log_probability_y = logtarget(y)

        α = min(1, exp(log_probability_y - log_probability_x))

        if α > rand()
            stats_accepted_values += 1
            x, y = y, x
        end

        # Step R3

        # This is taken from the second paragraph of section 5
        η = min(1, d * i^-γ)

        M = s.L * (I + η * (α-opt_α) * (u * u') / norm(u)^2 ) * s.L'
        # The paper has a proof that M is symmetric, so we declare that fact
        # to work around numerical rounding errors
        s = cholesky(Symmetric(M))

        output_chain[i, :] .= x
    end

    return (chain=output_chain, acceptance_rate=stats_accepted_values/n, S=s.L)
end

end
