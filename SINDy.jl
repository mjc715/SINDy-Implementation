using DifferentialEquations
using ForwardDiff
include("practice.jl")

# Define library functions here
f0(t, x, y, z) = 1
f1(t, x, y, z) = x(t)
f2(t, x, y, z) = x(t)^2
f3(t, x, y, z) = x(t)^3
f4(t, x, y, z) = y(t)
f5(t, x, y, z) = y(t)^2
f6(t, x, y, z) = x(t) * y(t)
f7(t, x, y, z) = exp(x(t))
f8(t, x, y, z) = z(t)
f9(t, x, y, z) = x(t) * z(t)

library = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

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
prob_vec = ODEProblem(lorenz!, [1.0, 1.0, 1.0], (0.0, 10.0))





function SINDy_implementation(prob_vec, library, lambda=0.25, n_points=100, n_iterations=10)

    # Solving ODEProblem and getting num of variables
    sol_vec = solve(prob_vec)
    vars = size(prob_vec.u0)[1]

    # Defining the variable functions
    if vars == 1
        x = (t) -> sol_vec(t)[1]
        y = (t) -> 0
        z = (t) -> 0
    elseif vars == 2
        x = (t) -> sol_vec(t)[1]
        y = (t) -> sol_vec(t)[2]
        z = (t) -> 0
    elseif vars == 3
        x = (t) -> sol_vec(t)[1]
        y = (t) -> sol_vec(t)[2]
        z = (t) -> sol_vec(t)[3]
    end

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
            row[i] = f(t, x, y, z)
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
    data_arrays = []

    if vars == 2
        midpoint = Integer(length(data_transform) / 2)
        row1 = data_transform[1:midpoint]
        row2 = data_transform[(midpoint+1):length(data_transform)]
        data_arrays = [[row1] [row2]]
    elseif vars == 1
        data_arrays = data_transform
    elseif vars == 3
        midpoint = Integer(length(data_transform) / 3)
        row1 = data_transform[1:midpoint]
        row2 = data_transform[(midpoint+1):(midpoint*2)]
        row3 = data_transform[(midpoint*2+1):length(data_transform)]
        data_arrays = [[row1] [row2] [row3]]
    end


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
        println(l, ": ", Xi)
    end

end


SINDy_implementation(prob_vec, library)