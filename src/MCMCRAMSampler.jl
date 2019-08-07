module MCMCRAMSampler

using LinearAlgebra, Random

function sample(logtarget, x0, s0, n; opt_α=0.234, γ=0.75)
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
        s = cholesky(Symmetric(M))

        output_chain[i, :] .= x
    end

    return output_chain
end

end
