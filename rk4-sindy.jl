using CairoMakie
using Optimization
using OptimizationOptimJL, OptimizationBBO

function XF(X, Φ, ξ, h)
    theta_1 = stack([f(X) for f in Φ])
    k1 = theta_1 * ξ
    X1_tilde = X + (h / 2) * k1

    theta_2 = stack([f(X1_tilde) for f in Φ])
    k2 = theta_2 * ξ
    X2_tilde = X + (h / 2) * k2

    theta_3 = stack([f(X2_tilde) for f in Φ])
    k3 = theta_3 * ξ
    X3_tilde = X + h * k3

    theta_4 = stack([f(X3_tilde) for f in Φ])
    k4 = theta_4 * ξ

    X_f = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return X_f
end

h = 0.01
x = [t^2 for t in range(0.0, 1.0, step=h)]
ξ0 = [2.0, 0.0]
α0 = 1.0
Φ = [
    x -> x,
    x -> x .^ 2] # list of functions

# X_f = XF(x[1:end-1], Φ, ξ0, h)
# @info "Simple error = $(sum(abs2.(X_f - x[2:end])))"

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, X_f)
lines!(ax, x[2:end])
fig

# minimize
# https://docs.sciml.ai/Optimization/stable/getting_started/
objective(ξ, α, X, Φ, h) = sum(abs2.(XF(X[1:end-1], Φ, ξ, h) - X[2:end])) + abs(α * sum(abs.(ξ)))
optfun(u, ps) = objective(u[1:end-1], u[end], ps[1], ps[2], ps[3])

u0 = [rand(), rand(), α0]
p = (x, Φ, h)

prob = OptimizationProblem(optfun, u0, p, lb = fill(-10.0, length(u0)), ub = fill(10.0, length(u0)))
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())

sol.objective # smallest value of optfun
sol.u # optimal [ξ1, ξ2, α]
