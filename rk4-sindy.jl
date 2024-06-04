using CairoMakie
using Optimization
using OptimizationOptimJL

h = 0.01
x = [t^2 for t in range(0.0, 1.0, step=h)]
X = x[1:end-1]
ξ = [0.0, 0.0]
α = 1.0
Φ = [
    x -> x,
    x -> x .^ 2] # list of functions


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

X_f = XF(X, Φ, ξ, h)

# println(X_f)
# println(X_f)
# println(x[2:end])

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, X_f)
lines!(ax, x[2:end])
fig

# minimize
# https://docs.sciml.ai/Optimization/stable/getting_started/
objective(ξ, α, Φ, h) = sum(abs2.(XF(x[1:end-1], Φ, ξ, h) - x[2:end])) + α * sum(abs.(ξ))
objective(u0, p) = objective(u0[1:end-1], u0[end], p[1], p[2])

u0 = [0.0, 0.0, α]
p = [Φ, h]

prob = OptimizationProblem(objective, u, p)
sol = solve(prob, NelderMead())

