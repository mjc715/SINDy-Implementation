x = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144] # trajectory
X = x[1:end-1]
ξ = stack([1.0, 0.0])
Φ = [
    x -> x,
    x -> x .^ 2] # list of functions


function XF(X, Φ, ξ)
    # RK4 applied to our f(θ, xi), starting at x[1]
    h = 1
    Θ1 = stack([f(X) for f in Φ])
    k1 = Θ1 * ξ
    X1_tilde = X + (h / 2) * k1

    Θ2 = stack([f(X1_tilde) for f in Φ])
    k2 = (Θ2 * ξ)
    X2_tilde = X1_tilde + (h / 2) * k2

    θ3 = stack([f(X2_tilde) for f in Φ])
    k3 = θ3 * ξ
    X3_tilde = X2_tilde + h * k3

    θ4 = stack([f(X3_tilde) for f in Φ])
    k4 = θ4 * ξ
    X4_tilde = X3_tilde + (h / 6) * k4

    Xf = X1_tilde * h / 6 + X2_tilde * h / 3 + X3_tilde * h / 3 + X4_tilde

    return Xf

end

println(XF(X, Φ, ξ))
println()
println(x[2:end])

# then compare XF to x[2:end]