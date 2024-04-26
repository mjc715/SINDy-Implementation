include(joinpath(@__DIR__, "..", "sindy2.jl"))

f0(t) = 1.0
f1(t) = 2*t
f2(t) = sin(t)
f3(t) = f1(t)*f2(t)
f4(t) = cos(t)
f5(t) = f1(t)*f4(t)

times = range(0.0, 10.0, length = 100) |> collect
target_data1 = [f1(t) + 2*f2(t) - 3*f3(t) + 0.1*rand() for t in times]
target_data2 = [4*f1(t) - 2*f4(t) - 5*f3(t) + 0.1*rand() for t in times]
target_data = [target_data1 ;; target_data2]

library = [f0, f1, f2, t -> f1(t)^1.2, f3, t -> f2(t)^2, f4, f5]
library_names = ["", "2t", "sin(t)", "(2t)^1.2", "2t*sin(t)", "sin(t)^2", "cos(t)", "2t*cos(t)"]
library_data = [f(t) for t in times, f in library]

si = sparse_representation(times, target_data, library_data, library_names = library_names, λ_sparse = 0.1, pretty_print = false)
si_pretty = sparse_representation(times, target_data, library_data, library_names = library_names, λ_sparse = 0.1, pretty_print = true)


###  bagging E-SINDy
n_bootstraps = 100
λ_sparse = 0.1
λ_ridge = 0.0
max_iters = 10
library_names = nothing
pretty_print = library_names !== nothing

Xi = zeros(size(target_data, 2), size(library_data, 2), n_bootstraps)
for b = 1:n_bootstraps
    idx = sample(1:length(times), length(times), replace = true)
    Xi_b = sparse_representation(times[idx], target_data[idx,:], library_data[idx,:], 
        λ_sparse = λ_sparse,
        λ_ridge = λ_ridge,
        max_iters = max_iters,
        library_names = library_names,
        pretty_print = pretty_print)

    Xi[:,:,b] .= stack(Xi_b, dims = 1)
end

ip = mean(map(x -> x != 0 ? 1.0 : 0.0, Xi), dims = 3)[:,:,1]




### library bagging
# n_bootstraps = 1000
# n_lib = floor(Int64, size(library_data, 2)/2)
# λ_sparse = 0.1
# λ_ridge = 0.0
# max_iters = 10
# library_names = nothing
# pretty_print = library_names !== nothing

# Xi = zeros(size(target_data, 2), size(library_data, 2), n_bootstraps)
# for b = 1:n_bootstraps
#     idx = sample(1:size(library_data, 2), n_lib, replace = false)
#     library_data_b = library_data[:,idx]
#     Xi_b = sparse_representation(times, target_data, library_data_b, 
#         λ_sparse = λ_sparse,
#         λ_ridge = λ_ridge,
#         max_iters = max_iters,
#         library_names = library_names,
#         pretty_print = pretty_print)

#     Xi[:,idx,b] .= stack(Xi_b, dims = 1)
# end

# ip = mean(map(x -> x > 0 ? 1.0 : 0.0, Xi), dims = 3)[:,:,1]