x = rand(10) # trajectory
X = x[1:end-1]
Φ = [
    x -> x, 
    x -> x .^ 2] # list of functions

function XF(X, Φ, ξ)
    # RK4 applied to our f(θ, xi), starting at x[1]
    Θ1 = stack([f(X) for f in Φ])
    k1 = Θ1 * ξ
    X1_tilde = X + (h/2)*k1

    Θ2 = stack([f(X1_tilde) for f in Φ])
    k2 = Θ2 * ξ

end

# then compare XF to x[2:end]