using DifferentialEquations
using ForwardDiff
include("practice.jl")

function f!(du, u, p, t)
    du[1] = 2t + t^2
end

prob = ODEProblem(f!, [0.0], (0.0, 10.0))
sol = solve(prob) # function you can evaluate, like sol(3.0)

t_final = 1.0
times = range(0.0, t_final, length=100) |> collect
data = [ForwardDiff.derivative(t -> sol(t)[1], ti) for ti in times]

## multivariable, vector output
function f_vec!(du, u, p, t)
    du[1] = u[1] - u[2]^2
    du[2] = u[1] * u[2] + 3
end


prob_vec = ODEProblem(f_vec!, [0.0, 0.0], (0.0, t_final))
sol_vec = solve(prob_vec) # function you can evaluate, like sol(3.0)

x(t) = sol_vec(t)[1]
y(t) = sol_vec(t)[2]
f0(t) = 1
f1(t) = x(t)
f2(t) = x(t)^2
f3(t) = x(t)^3
f4(t) = y(t)
f6(t) = y(t)^2
f5(t) = x(t) * y(t)
f7(t) = exp(x(t))

data = [ForwardDiff.derivative(t -> sol_vec(t), ti) for ti in times]
# data1 = [f2(t) - f1(t) + 0.1 * rand() for t in times] # 1. x^2-x
# data2 = [f7(t) - 3 + 0.1 * rand() for t in times]
# data = [data1; data2] # 2. xy - 3 (x = t, y = sin(t))
library = [f0, f1, f2, f4, f5, f6, f7]
vars = 2
data_transform = zeros(size(data[1])[1], size(data)[1])

for i in 1:size(data_transform)[1]
    for j in 1:size(data_transform)[2]
        data_transform[i, j] = data[j][i]
    end
end


Xi = sparse_representation(library, data_transform, times, vars)

# SINDy(prob_vec, library, lambda = 0.25, n_points = 100)
# f(x) = x^2 -> x(t) = sol_vec(t)[1]
# f_new(t) = f(x(t))
# Also convert data -> prob_vec