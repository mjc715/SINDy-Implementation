using DifferentialEquations
using Interpolations
import MultivariateStats
using Markdown

"""
    sparse_representation(times, target_data, library_data; λ_sparse, λ_ridge, max_iters, library_names, pretty_print)
"""
function sparse_representation(
    times::Vector{<:Real},
    target_data::Vector{<:Real},
    library_data::Matrix{<:Real};
    λ_sparse::Real = 0.1,
    λ_ridge::Real = 0.0, 
    max_iters::Integer = 10,
    library_names::Union{Vector{String}, Nothing} = nothing, 
    pretty_print::Bool = library_names !== nothing)

    @assert all([length(times) > 0, length(target_data) > 0, length(library) > 0])
    @assert all(λ_sparse .> 0)
    @assert max_iters > 0
    @assert length(times) == length(target_data)
    @assert length(times) == size(library_data, 1)

    Xi = MultivariateStats.ridge(library_data, target_data, λ_ridge, bias = false)
    
    smallinds = findall(p -> abs(p) < λ_sparse, Xi)
    biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
    for _ in 1:max_iters
        smallinds_next = findall(p -> abs(p) < λ_sparse, Xi) # library functions with small coefficients
    
        if smallinds_next == smallinds
            # this iteration has not changed the location of smallinds, so exit the loop
            break
        else
            smallinds = smallinds_next
        end
    
        Xi[smallinds] .= 0
        biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
        Xi[biginds] = MultivariateStats.ridge(library_data[:, biginds], target_data, λ_ridge, bias = false)
    end
    
    if pretty_print && library_names !== nothing
        nums = string.(round.(Xi[biginds], sigdigits = 3)) .|> x -> "[$x]"
        output_string = ""
        for i = 1:length(nums)
            output_string *= nums[i] * library_names[biginds[i]]
            output_string *= i == length(nums) ? "" : " + "
        end
    
        return output_string
    else
        return Xi
    end
end

function sparse_representation(
    times::Vector{<:Real},
    target_data::Matrix{<:Real},
    library_data::Matrix{<:Real};
    λ_sparse::Real = 0.1, 
    λ_ridge::Real = 0.0, 
    max_iters::Integer = 10,
    library_names::Union{Vector{String}, Nothing} = nothing, 
    pretty_print::Bool = library_names !== nothing)

    @assert size(target_data, 1) == length(times)

    Xi = []
    for i = 1:size(target_data, 2)
        sr = sparse_representation(times, target_data[:, i], library_data, 
            λ_sparse = λ_sparse, 
            λ_ridge = λ_ridge,
            max_iters = max_iters, 
            library_names = library_names, 
            pretty_print = pretty_print)
        push!(Xi, sr)
    end

    return Xi
end

function process_trajectories(
    times::Vector{<:Real}, 
    trajectories::Matrix{<:Real}; 
    n_times::Union{Integer, Nothing} = nothing)

    @assert issorted(times)
    @assert length(times) == size(trajectories, 1)

    itps_gridded = [interpolate((times,), trajectories[:,i], Gridded(Linear())) for i = 1:size(trajectories, 2)]

    ts = range(times[1], times[end], length = n_times === nothing ? length(times) : n_times)
    itps = [cubic_spline_interpolation(ts, itps_gridded[i].(ts)) for i = 1:size(trajectories, 2)]

    return itps
end

function sindy(
    times::Vector{<:Real},
    trajectories::Matrix{<:Real},
    library::Vector{<:Function}; 
    order::Integer = 1,
    λ_sparse::Real = 0.1, 
    λ_ridge::Real = 0.0, 
    max_iters::Integer = 10,
    library_names::Union{Vector{String}, Nothing} = nothing, 
    pretty_print::Bool = true)
    
    @assert order in [1, 2]

    itps = process_trajectories(times, trajectories)
    if order == 1 # first order ODE, trying to match d/dt
        target_data = [Interpolations.gradient(itps[i], t)[1] for t in times, i = 1:size(trajectories, 2)]
    elseif order == 2 # second order ODE, trying to match d^2/dt^2
        # ned to provide access to d/dt
        # e.g. if have two variables, then itps = [x(t), y(t), dxdt(t), dydt(t)]
        itps = [itps ; [t -> Interpolations.gradient(itps[i], t)[1] for i = 1:length(itps)]] 
        target_data = [Interpolations.hessian(itps[i], t)[1] for t in times, i = 1:size(trajectories, 2)]
    end

    library_data = [f(t, itps) for t in times, f in library] # N_times x N_library_funcs matrix
    
    return sparse_representation(times, target_data, library_data, 
        λ_sparse = λ_sparse, 
        λ_ridge = λ_ridge,
        max_iters = max_iters, 
        library_names = library_names,
        pretty_print = pretty_print)
end