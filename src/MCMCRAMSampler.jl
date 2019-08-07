module MCMCRAMSampler

using LinearAlgebra, Random

function sample(logtarget, x0::AbstractVector{<:Number}, s0::AbstractVector{<:Number}, n::Int; opt_α=0.234, γ=0.75)
    length(x0) == length(s0) || error("x0 and s0 must have the same length.")
    n > 0 || error("n must be larger than 0.")
    0 < opt_α < 1 || error("opt_α must be between 0 and 1.")
    0.5 < γ <= 1 || error("γ must be between 0.5 and 1.")

    x = copy(x0)
    s = cholesky(s0) # Probably wrong

    par_count = length(x0)

    u = zeros(par_count)
    y = zeros(par_count)

    log_probability_x = logtarget(x0)

    output_chain = Matrix{Float64}(undef, n, par_count)

    for i in 1:n
        # Step R1
        randn!(u)
        y[:] .= x .+ s.L * u

        # Step R2
        log_probability_y = logtarget(y)

        α = min(1, exp(log_probability_y - log_probability_x))

        if α > rand()
            x, y = y, x
        end

        # Step R3

        η = i^-γ
        M = s.L * (I + η * (α-opt_α) * (u * u') / norm(u)^2 ) * s.L'
        # The paper has a proof that M is symmetric, so we declare that fact
        # to work around numerical rounding errors
        s = cholesky(Symmetric(M))

        output_chain[i, :] .= x
    end

    return output_chain
end

end
