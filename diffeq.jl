using DifferentialEquations
using ForwardDiff

function f!(du, u, p, t)
    du[1] = 2t + t^2
end

prob = ODEProblem(f!, [0.0], (0.0, 10.0))
sol = solve(prob) # function you can evaluate, like sol(3.0)

times = range(0.0, 10.0, length = 5) |> collect
data = [ForwardDiff.derivative(t -> sol(t)[1], ti) for ti in times]

### multivariable, vector output
function f_vec!(du, u, p, t)
    du[1] = 2u[1] - u[2]^2
    du[2] = u[1]*u[2] + 3
end

prob_vec = ODEProblem(f_vec!, [0.0, 0.0], (0.0, 1.0))
sol_vec = solve(prob_vec) # function you can evaluate, like sol(3.0)

times = range(0.0, 1.0, length = 5) |> collect
data = [ForwardDiff.derivative(t -> sol_vec(t), ti) for ti in times]