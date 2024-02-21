# Sparse representation algorithm
using DataDrivenDiffEq, ModelingToolkit, DataDrivenSparse, LinearAlgebra, OrdinaryDiffEq, StableRNGs

@variables u t

function example1(u, p, t)
    x, y = u
    dx = 2 * y - 0.1 * x
    dy = -2 * x - 0.1 * y
    return [dx, dy]
end

function optimization()
    lambda = 1 # sparcification parameter
    n = 10
    dXdt = 10
    Xi = theta / dXdt

    for k in 1:10
        smallinds = abs(Xi) < lambda
        Xi[smallinds] = 0
        for j in 1:n
            biginds = ~smallinds[:, j]
            Xi[biginds, j] = theta[:, biginds] / dXdt[:, ind]
        end
    end
end

u0 = [1.0; 0.0]
tspan = (0.0, 10.0)
dt = 0.1

prob = ODEProblem(example1, u0, tspan)
sol = solve(prob, Tsit5(), saveat=dt)

X = sol[:, :] + 0.1 .* randn(StableRNG(10), size(sol))
ts = sol.ts
prob = ContinuousDataDrivenProblem(X, ts, GaussianKernel(),)
opt = STLSQ(1)

h = Num[polynomial_basis(u, 4)]
theta = Basis(h, u) # basis functions

