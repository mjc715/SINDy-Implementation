using DifferentialEquations
using ForwardDiff
include("data/traj.jl")
# ] > activate . > up

# Define library functions here
f0(t, sol) = 1
f1(t, sol) = sol(t)[1]
f2(t, sol) = sol(t)[1]^2
f3(t, sol) = sol(t)[1]^3
f4(t, sol) = sol(t)[2]
f5(t, sol) = sol(t)[2]^2
f6(t, sol) = sol(t)[1] * sol(t)[2]
f7(t, sol) = exp(sol(t)[1])
f8(t, sol) = vx(sol(t)[1], sol(t)[2], t)
f9(t, sol) = vy(sol(t)[1], sol(t)[2], t)
f10(t, sol) = sol(t)[3]
f11(t, sol) = sol(t)[4]
f12(t, sol) = sol(t)[1] * sol(t)[3]

func_names = ["", "x", "x^2", "x^3", "y", "y^2", "xy", "e^x", "vx", "vy", "dx", "dy", "xz"]
library = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

# func_names = ["", "x", "x^2", "x^3", "y", "y^2", "xy", "e^x", "vx", "vy"]
# library = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

# func_names = ["dx", "dy", "vx", "vy"]
# library = [f10, f11, f8, f9]

function f_vec!(du, u, p, t)
    du[1] = u[1] - u[2]^2
    du[2] = u[1] * u[2] + 3
end

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

# Creating ODEProblem to solve
# prob_vec = ODEProblem(f_vec!, [1.0, 1.0], (0.0, 10.0))
# prob_vec = ODEProblem(lorenz!, [1.0, 1.0, 1.0], (0.0, 10.0))
prob_vec = generate_trajectory("full")


function SINDy_implementation(prob_vec, library; lambda=0.1, n_points=100, n_iterations=10)
    # n_points = 50
    # n_iterations = 10
    # lambda = 0.1
    # Solving ODEProblem and getting num of variables
    sol_vec = solve(prob_vec)
    u0 = vec(prob_vec.u0)
    vars = size(u0)[1]

    # Getting time points and data from them
    times = range(prob_vec.tspan[1], prob_vec.tspan[2], length=n_points) |> collect
    data = [ForwardDiff.derivative(t -> sol_vec(t), ti) for ti in times]

    data_transform = zeros(size(data[1])[1], size(data)[1])

    # Formatting data for use in sparse representation
    for i in 1:size(data_transform)[1]
        for j in 1:size(data_transform)[2]
            data_transform[i, j] = data[j][i]
        end
    end

    helper = []

    # Creating theta using helper array
    for f in library
        i = 1
        row = zeros(length(times))
        for t in times
            row[i] = f(t, sol_vec)
            i += 1
        end
        push!(helper, row)

    end
    theta = zeros(length(times), length(library))
    dims = size(theta)

    for c in 1:dims[2]
        for r in 1:dims[1]
            theta[r, c] = helper[c][r]
        end
    end



    # More data formatting
    data_transform = vec(permutedims(data_transform))
    data_arrays = reshape(data_transform, Integer(length(data_transform) / vars), vars)
    data_arrays = [data_arrays[:, i] for i = 1:size(data_arrays, 2)]



    # Sparse representation algorithm
    for l in 1:vars
        Xi = theta \ data_arrays[l]
        for k in 1:n_iterations
            smallinds = findall(p -> (abs(p) < abs(lambda)), Xi) #array of indicies with small coefficients
            Xi[smallinds] .= 0
            biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
            Xi[biginds] = theta[:, biginds] \ data_arrays[l]
        end
        # Print results for each equation
        line = "$l: "
        for (num, name) in zip(Xi, func_names)
            # println(num)
            if abs(num) > 10e-6
                line *= "$num$name + "
            end
        end
        line = line[1:length(line)-3]
        println(line)

    end

end


SINDy_implementation(prob_vec, library, n_points=100, lambda=0.05)

# Get "full" to work 4 vars
# implement for generic # of vars

# Data transform not properly getting all vars