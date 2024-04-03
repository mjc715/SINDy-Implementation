using DifferentialEquations
using ForwardDiff

f1(t) = t
f2(t) = t^2
f3(t) = t^3

function f!(du, u, p, t)
    du[1] = 2t + t^2
end

prob = ODEProblem(f!, [0.0], (0.0, 10.0))
sol = solve(prob) # function you can evaluate, like sol(3.0)

times = range(0.0, 1.0, length=10) |> collect
data = [ForwardDiff.derivative(t -> sol(t)[1], ti) for ti in times]

### multivariable, vector output
function f_vec!(du, u, p, t)
    du[1] = 2u[1] - u[2]^2
    du[2] = u[1] * u[2] + 3
end

prob_vec = ODEProblem(f_vec!, [0.0, 0.0], (0.0, 1.0))
sol_vec = solve(prob_vec) # function you can evaluate, like sol(3.0)

data = [ForwardDiff.derivative(t -> sol_vec(t), ti) for ti in times]

times = [times; times]

n_iterations = 10
row_1 = [1 for t in times]
row_x = [f1(t) for t in times]
row_y = [f1(2t) for t in times]
row_xy = [f1(t) * f1(2t) for t in times]
row_y2 = [f2(2t) for t in times]
theta = [row_1;; row_x;; row_y;; row_xy;; row_y2]
row3 = theta[1:10, :]
row4 = theta[11:20, :]
thetas = [[row3] [row4]]
data1 = [data[i][1] for i in 1:length(data)]
data2 = [data[i][2] for i in 1:length(data)]
datas = [data1, data2]
lambda = 0.25
for l in 1:size(datas, 1)
    Xi = thetas[l] \ datas[l]
    for k in 1:n_iterations
        smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
        # println(smallinds)
        Xi[smallinds] .= 0
        biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
        # println(biginds)
        Xi[biginds] = thetas[l][:, biginds] \ datas[l]
    end
    println(l, ": ", Xi)
end