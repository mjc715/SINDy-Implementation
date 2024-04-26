include(joinpath(@__DIR__, "..", "sindy2.jl"))

f0(t) = 1.0
f1(t) = 2*t
f2(t) = sin(t)
f3(t) = f1(t)*f2(t)
f4(t) = cos(t)
f5(t) = f1(t)*f4(t)

times = range(0.0, 10.0, length = 100) |> collect
target_data1 = [f1(t) + 2*f2(t) - 3*f1(t)*f2(t) + 0.01*rand() for t in times]
target_data2 = [4*f1(t) - 2*f4(t) - 5*f1(t)*f2(t) + 0.01*rand() for t in times]
target_data = [target_data1 ;; target_data2]

library = [f0, f1, f2, t -> f1(t)^1.2, f3, t -> f2(t)^2, f4, f5]
library_names = ["", "2t", "sin(t)", "(2t)^1.2", "2t*sin(t)", "sin(t)^2", "cos(t)", "2t*cos(t)"]
library_data = [f(t) for t in times, f in library]

sparse_representation(times, target_data, library_data, library_names = library_names, λ_sparse = 0.1, pretty_print = true)