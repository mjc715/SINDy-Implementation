### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 8976a397-bbaa-4439-b19a-09dc87258f45
begin
	using DifferentialEquations
	using BSplineKit
	using MultivariateStats: llsq, ridge, mean
	using Random: seed!
	using StatsBase: sample
	using Combinatorics: multiexponents
	using Markdown
	using Latexify
	using CairoMakie; set_theme!(theme_latexfonts())
	using MAT
	using Interpolations: cubic_spline_interpolation
	using SargassumColors
end

# ╔═╡ 5725e2ce-40ec-42ee-b521-2c54922bad7f
md"""
# Front Matter
"""

# ╔═╡ 5fad43f1-80e3-43ae-99bf-daf288b27e4f
md"""
# Sparse Representation
"""

# ╔═╡ ec34c8de-62fc-4f0d-b05d-43423d09d823
struct STRidgeOptions
	lambda_sparse::Float64
	lambda_ridge::Float64
	max_iters::Int64

	function STRidgeOptions(; lambda_sparse::Real = 0.1, lambda_ridge::Real = 0.1, max_iters::Integer = 10)
		lambda_sparse = convert(Float64, lambda_sparse)
		lambda_ridge = convert(Float64, lambda_ridge)
		max_iters = convert(Int64, max_iters)

		return new(lambda_sparse, lambda_ridge, max_iters)
	end
end

# ╔═╡ 877ef8d8-2499-4692-a26a-b345efbd597a
function STRidge(
	target_data::Vector{<:Real}, 
	library_data::Matrix{<:Real}; # N_times x N_vars
	options::STRidgeOptions = STRidgeOptions())

	lambda_sparse, lambda_ridge, max_iters = options.lambda_sparse, options.lambda_ridge, options.max_iters
	
	rr(data) = ridge(data, target_data, lambda_ridge, bias = false) 
	Xi = rr(library_data)
	
	for _ in 1:max_iters
	  smallinds = findall(p -> abs(p) < lambda_sparse, Xi)
	  Xi[smallinds] .= 0
	  biginds = setdiff(1:length(Xi), smallinds)
	  Xi[biginds] = rr(library_data[:, biginds])
	end

  return Xi # N_vars
end

# ╔═╡ e6db750f-73bc-4f4a-b6d2-ff27a575d694
function STRidge(
	target_data::Matrix{<:Real}, 
	library_data::Matrix{<:Real}; 
	options::STRidgeOptions = STRidgeOptions())

    Xi = zeros(size(library_data, 2), size(target_data, 2))
    for i = 1:size(target_data, 2)
        Xi[:,i] .= STRidge(target_data[:, i], library_data, options = options) 
    end

    return Xi # N_vars x N_targets
end

# ╔═╡ 831a346a-3200-497f-87c3-f6bca5d2e146
function calculate_derivatives(
  times::Vector{<:Real},
  trajectories::Matrix{<:Real},
  derivative_order::Integer;
  spline_order::Integer = max(derivative_order + 1, 4))

  @assert issorted(times)
  @assert derivative_order >= 0

  bo = BSplineOrder(spline_order)
  bn = BSplineKit.Natural()

  itps = [spline(interpolate(times, trajectories[:,i], bo, bn))
    for i = 1:size(trajectories, 2)]

  return Dict(i => [diff(S, Derivative(i))(t) for t in times, S in itps] for i = 0:derivative_order)
end

# ╔═╡ aa630bbe-1447-4a89-b32f-86911d35661c
begin
	struct LibFun{F<:Function}
		f::F # should return a number
		name::String
	
		function LibFun(f::Function, name::Union{String, Nothing} = nothing)
			return new{typeof(f)}(f, name === nothing ? "?" : name)
		end
	end
	
	function (lf::LibFun)(t::Real, u::Vector{<:Real})
		return lf.f(t, u)
	end

	function (lf::LibFun)(t::Vector{<:Real}, u::Vector{<:Vector{<:Real}})
		@assert all(length(t) == length(u[i]) for i = 1:length(u))
		return [lf(t[i], [u[j][i] for j = 1:length(u)]) for i = 1:length(t)]
	end
end

# ╔═╡ 4621b7dd-530e-4488-8952-8581195ec976
mutable struct SparseDynamicsProblem
	t::Vector{Float64}
	traj::Matrix{Float64}
	order::Int64
	var_data::Vector{Vector{Float64}} # [var1.(t), var2.(t), ...]
	var_names::Vector{String} # length(var_names) == length(var_data)
	var_functions::Vector{LibFun}
	target::Matrix{Float64}
	library_data::Vector{Vector{Float64}}
	library_functions::Vector{LibFun}
	
	function SparseDynamicsProblem(
		t::Vector{<:Real}, 
		traj::Matrix{<:Real}, 
		order::Integer; 
		var_names::Union{Nothing, Vector{String}} = nothing)

		@assert order >= 1
		n_vars = size(traj, 2)
		
		t = convert(Vector{Float64}, t)
		traj = convert(Matrix{Float64}, traj)
		order = convert(Int64, order)

		ds = calculate_derivatives(t, traj, order) # dict
		var_data = hcat([ds[key] for key = 0:order-1]...)
		var_data = [var_data[:,i] for i = 1:size(var_data, 2)]
		target = ds[order]

		var_names = var_names === nothing ? ["x$(i)" for i = 1:n_vars] : var_names
		@assert length(var_names) == n_vars

		for j = 1:order-1, i = 1:n_vars
			push!(var_names, "∂$(j == 1 ? "" : "^$(j)")$(var_names[i])")
		end

		var_funcs = [LibFun((t, u) -> u[i], var_names[i]) for i = 1:length(var_names)]
		
		return new(t, traj, order, var_data, var_names, var_funcs, target, Vector{Float64}[], LibFun[])
	end
end

# ╔═╡ 0eca8a7c-ec2a-4b97-a8cf-65282bc5b7e0
function add_library_function!(sdp::SparseDynamicsProblem, library_function::LibFun)
	push!(sdp.library_data, library_function(sdp.t, sdp.var_data))
	push!(sdp.library_functions, library_function)
	return nothing
end

# ╔═╡ ee9f84a0-a949-45b4-81d4-e8ac73236420
function add_library_function!(sdp::SparseDynamicsProblem, library_function::Vector{LibFun})
	for lib_fun in library_function
		add_library_function!(sdp, lib_fun)
	end

	return nothing
end

# ╔═╡ 74742e45-c5d5-45a1-b161-f63ab82bd0d7
function polynomials(
  lib_funcs::Vector{LibFun},
  poly_order::Integer)

  @assert length(lib_funcs) >= 1
  @assert poly_order >= 0

  n_vars = length(lib_funcs)
  funcs = LibFun[]

  for i = 0:poly_order, ex in multiexponents(n_vars, i)
    name = "" 
    for i = 1:n_vars
      if ex[i] != 0
		  if ex[i] == 1
			  name *= lib_funcs[i].name
		  else
			  name *= "[$(lib_funcs[i].name)]^$(ex[i])"
		  end
      end
    end

	fn = LibFun(
		(t, u) -> prod(lib_funcs[i](t, u)^ex[i] for i = 1:n_vars), 
		name)
	  
    push!(funcs, fn)
  end

  return funcs
end

# ╔═╡ 7532911c-da81-4ab9-92da-affa677d6e27
struct SparseDynamicsResult{VM, F}
	coeffs::VM
	rmse::Vector{Float64}
	names::Vector{String}
	odefun::F

	function SparseDynamicsResult(sdp::SparseDynamicsProblem, coeffs::VM) where {VM}
		n_libs = size(coeffs, 1)
		n_targets = size(coeffs, 2)

		rmse = [sqrt(mean(abs2.(sdp.target[:,target_idx] - sum(sdp.library_data .* coeffs[:,target_idx])))) for target_idx = 1:n_targets]
		
		f = [(t, u) -> sum(
			coeffs[lib_idx, target_idx]*sdp.library_functions[lib_idx](t, u) 
				for lib_idx = 1:n_libs)
		for target_idx = 1:n_targets]

		names = String[]
		for target_idx = 1:n_targets
			name = ""
			for lib_idx = 1:n_libs
				c = coeffs[lib_idx, target_idx] 
				if !(c ≈ 0.0)
					name *= "$(round(c, sigdigits = 3)) $(sdp.library_functions[lib_idx].name) + "
				end
			end

			name = name[1:end-3]
			push!(names, name)
		end

		n_vars = size(sdp.traj, 2)
		order = sdp.order

		if order == 1
			function odefun1!(du, u, p, t)
				for i = 1:n_vars
					du[i] = f[i](t, u)
				end
			end
		
			return new{VM, typeof(odefun1!)}(coeffs, rmse, names, odefun1!)
		else
		    n_du = n_vars*order
		    idx_v = 1:n_du-n_vars
		    idx_f = n_du-n_vars+1:n_du
		
		    function odefunN!(du, u, p, t)
		      for i in idx_v
		        du[i] = u[i + n_vars]
		      end
		
		      for i = 1:length(idx_f)
		        du[idx_f[i]] = f[i](t, u)
		      end
		    end
		
			return new{VM, typeof(odefunN!)}(coeffs, rmse, names, odefunN!)
		end
	end
end

# ╔═╡ 1b3ed7d7-bb7c-4faf-ab89-5c65f816578b
function STRidge(sdp::SparseDynamicsProblem; options::STRidgeOptions = STRidgeOptions())
	coeffs = STRidge(sdp.target, stack(sdp.library_data), options = options)
	return SparseDynamicsResult(sdp, coeffs)
end

# ╔═╡ 7388ca53-d7aa-4e95-a830-c99381550413
md"""
# Lorenz
"""

# ╔═╡ 0d4d2c16-0669-451b-99ee-8ea93c172f87
begin
	ics_lorenz = [1.0, 1.0, 1.0] 		
	tspan_lorenz_test = (0.0, 20.0)
	n_times_lorenz = 300 	
	tspan_lorenz_train = (0.0, 10.0)
	nothing
end

# ╔═╡ 5b5eb1dd-ab3e-4f6e-8189-c123c6505252
begin
	function lorenz!(du, u, p, t)
	    x, y, z = u
	    du[1] = 10.0 * (y - x)
	    du[2] = x * (28.0 - z) - y
	    du[3] = x * y - (8 / 3) * z
	end

	sol_lorenz = ODEProblem(lorenz!, ics_lorenz, tspan_lorenz_test) |> solve
	
	times_lorenz_train = range(tspan_lorenz_train[1], tspan_lorenz_train[2], length = n_times_lorenz) |> collect
	traj_lorenz_train = [sol_lorenz(t)[i] for t in times_lorenz_train, i = 1:length(ics_lorenz)]
	
	times_lorenz_test = range(tspan_lorenz_test[1], tspan_lorenz_test[2], length = floor(Int64, (tspan_lorenz_test[2]/tspan_lorenz_train[2])*n_times_lorenz)) |> collect
	traj_lorenz_test = [sol_lorenz(t)[i] for t in times_lorenz_test, i = 1:length(ics_lorenz)]
	
	nothing
end

# ╔═╡ 22a674b3-5fa3-4cfe-b854-d3fef37e6f02
begin
	sdp_lorenz = SparseDynamicsProblem(times_lorenz_train, traj_lorenz_train, 1, var_names = ["x", "y", "z"])
	
	polys_lorenz = polynomials(sdp_lorenz.var_functions, 2)
	add_library_function!(sdp_lorenz, polys_lorenz)
	
	sdr_lorenz = STRidge(sdp_lorenz)

	sol_lorenz_sindy = ODEProblem(sdr_lorenz.odefun, ics_lorenz, tspan_lorenz_test) |> solve
	traj_lorenz_sindy = [sol_lorenz_sindy(t)[i] for t in times_lorenz_test, i = 1:length(ics_lorenz)]
	nothing
end

# ╔═╡ 634cbbb0-a1eb-4b4c-ad19-1bc4e56b5ae8
sdr_lorenz.names

# ╔═╡ 35a4d5d1-a490-4853-a141-48bbc8a194da
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	labels = [L"x", L"y", L"z"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	for i = 1:size(traj_lorenz_test, 2)
	  lines!(ax, times_lorenz_test, traj_lorenz_test[:,i], label = labels[i])
	end
	
	for i = 1:size(traj_lorenz_sindy, 2)
	  lines!(ax, times_lorenz_test, traj_lorenz_sindy[:,i], linestyle = :dash, label = labels_sindy[i])
	end
	
	axislegend(ax)

	fig
end

# ╔═╡ be9c7a59-9a2a-41d8-8f63-d4919b6cee4e
md"""
# Gulf Stream Data
"""

# ╔═╡ 31f79ec2-90ec-4381-b112-f682f01e66c9
md"""
## Loading Data and Interpolants
"""

# ╔═╡ 5d04a43f-98d8-475c-97a5-17c0db2a259c
datapath = joinpath(pwd(), "..", "data") |> abspath

# ╔═╡ 77c2ea65-cd7b-45ce-b168-7f4e5280628b
md"""
## Fluid
"""

# ╔═╡ 0e51b25d-55d0-4c0d-b2fd-3e951329266f
ics_fluid = [-700, -700] .+ (3*rand(), 3*rand())

# ╔═╡ 31b759e3-60fc-4c9a-aa7c-053906125d0b
md"""
## Slow
"""

# ╔═╡ 7ffdd06a-1462-45a7-8b42-836db4a93aad
ics_slow = [-700, -700] .+ (3*rand(), 3*rand())

# ╔═╡ 8d8e2bdc-c35e-4332-884a-34bc49707ffa
md"""
## Full
"""

# ╔═╡ 49483d2c-a284-4643-8000-ef5d664b4e18
begin
	ics_full = [-700, -700] .+ (3*rand(), 3*rand())
	ics_full_v = [0, 0] .+ (rand(), rand())
end

# ╔═╡ 58b47ca8-fa19-44f0-922a-58ac71816eb4
md"""
# Extra
"""

# ╔═╡ 788f628e-c77c-49a5-ac81-a527ede9cfbd
md"""
# Utilities
"""

# ╔═╡ 6ca5c925-621e-4d6f-a379-e20d4f7bc225
begin
@info "Maximizing notebook width."
	
html"""
<style>
main {
	max-width: calc(100vw - 80px);
	align-self: flex-start !important;
	margin-left: 30px !important;
}
</style>
"""
end

# ╔═╡ 840cada2-6d5f-4bad-bee7-5faba2071a08
const EARTH_RADIUS = 6371.0 

# ╔═╡ 40d8c53e-c18d-48d5-93d2-b5afb42ab962
struct EquirectangularReference{T<:AbstractFloat}
    lon0::T
    lat0::T
    R::T

    function EquirectangularReference(; lon0::Real = -75.0, lat0::Real = 10.0, R::Real = EARTH_RADIUS)
        @assert -180.0 <= lon0 <= 180.0 "The longitude must be between -180 degrees and 180 degrees."
        @assert -90 <= lat0 <= 90 "The latitude must be between -90 degrees and 90 degrees."
    
        lon0, lat0, R = promote(float(lon0), float(lat0), float(R))
    
        return new{typeof(lon0)}(lon0, lat0, R)
    end
end

# ╔═╡ 3bd280c2-e6c3-411e-a4e8-5969afce0dd7
let
	xyt0 = matread(joinpath(datapath, "sargassum-xyorigin.mat"))
	lon_origin = xyt0["lon_origin"]
	lat_origin = xyt0["lat_origin"]
	global sarg_ref = EquirectangularReference(lon0=lon_origin, lat0=lat_origin)
	
	xyt = matread(joinpath(datapath, "sargassum-xyt.mat"))
	x, y, t = (xyt["x"], xyt["y"], xyt["t"]) .|> vec
	x = range(x[1], x[end], length=length(x))
	y = range(y[1], y[end], length=length(y))
	t = range(0, t[end] - t[1], length=length(t))
	
	uv = matread(joinpath(datapath, "sargassum-uv.mat"))
	vx_data, vy_data = (uv["u"], uv["v"]) .|> x -> permutedims(x, (2, 1, 3))

	global sarg_times = deepcopy(t)
	global vx = cubic_spline_interpolation((x, y, t), vx_data)
	global vy = cubic_spline_interpolation((x, y, t), vy_data)

	global function f_fluid!(du, u, p, t)
	    x, y = u
	    du[1] = vx(x, y, t)
	    du[2] = vy(x, y, t)
	    return nothing
	end

	f = 2 * 2 * π * sin(lat_origin * π / 180)
	δ = 0.9
	τ = 1 / f
	
	global function f_slow!(du, u, p, t)
	    x, y = u
	    du[1] = vx(x, y, t) + f * τ * (1 - δ) * vy(x, y, t)
	    du[2] = vy(x, y, t) - f * τ * (1 - δ) * vx(x, y, t)
	    return nothing
	end

	global function f_full!(du, u, p, t)
	    x, y, dxdt, dydt = u
	    du[1] = dxdt
	    du[2] = dydt
	    du[3] = f * (dydt - δ * vy(x, y, t)) + (vx(x, y, t) - dxdt) / τ
	    du[4] = f * (δ * vx(x, y, t) - dxdt) + (vy(x, y, t) - dydt) / τ
	    return nothing
	end

	nothing
end

# ╔═╡ bb5174ab-523e-4363-ba47-38d9c47f0de4
begin
	tspan_fluid_test = (0.0, 120.0)
	n_times_fluid = 1000 	
	tspan_fluid_train = (0.0, 90.0)

	###

	sol_fluid = ODEProblem(f_fluid!, ics_fluid, tspan_fluid_test) |> x -> solve(x, Tsit5())
	
	times_fluid_train = range(tspan_fluid_train[1], tspan_fluid_train[2], length = n_times_fluid) |> collect
	traj_fluid_train = [sol_fluid(t)[i] for t in times_fluid_train, i = 1:length(ics_fluid)]
	
	times_fluid_test = range(tspan_fluid_test[1], tspan_fluid_test[2], length = floor(Int64, (tspan_fluid_test[2]/tspan_fluid_train[2])*n_times_fluid)) |> collect
	traj_fluid_test = [sol_fluid(t)[i] for t in times_fluid_test, i = 1:length(ics_fluid)]
	
	nothing
end

# ╔═╡ cd6228f3-8b1c-40e1-bda4-b62de5d9b483
let
	global cutoffs_fluid = []
	global sdrs_fluid = []
	n_terms_poss = []
	for lambda_sparse in 10 .^ range(-3, 1, length = 100)
	  sdp_fluid = SparseDynamicsProblem(times_fluid_train, traj_fluid_train, 1, var_names = ["x", "y"])
	
	  x_fluid, y_fluid = sdp_fluid.var_functions
	  vx_fluid = LibFun((t, u) -> vx(x_fluid(t, u), y_fluid(t, u), t), "vx")
	  vy_fluid = LibFun((t, u) -> vy(x_fluid(t, u), y_fluid(t, u), t), "vy") 
	  lib_funs_fluid = [x_fluid, y_fluid, vx_fluid, vy_fluid]
		
	  polys_fluid = polynomials(lib_funs_fluid, 2)
	  add_library_function!(sdp_fluid, polys_fluid)
	
      options_fluid = STRidgeOptions(lambda_sparse = lambda_sparse, lambda_ridge = 0.1, max_iters = 10)
	  sdr_fluid = STRidge(sdp_fluid, options = options_fluid)
		
	  rmse = sdr_fluid.rmse |> x -> sum(x)/length(x)
	  n_terms = count(iszero, sdr_fluid.coeffs)
	
	  if !(n_terms in n_terms_poss)
	    push!(cutoffs_fluid, [lambda_sparse, n_terms, rmse])
	    push!(sdrs_fluid, sdr_fluid)
	    push!(n_terms_poss, n_terms)
	  else
	    cutoffs_fluid[end] = [lambda_sparse, n_terms, rmse]
	    sdrs_fluid[end] = sdr_fluid
	  end
	end
	cutoffs_fluid = stack(cutoffs_fluid, dims = 1)
	
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = L"\log10(\lambda)", ylabel = "RMSE")
	scatter!(ax, log10.(cutoffs_fluid[:,1]), cutoffs_fluid[:,3], color = :black, markersize = 20)
	fig
end

# ╔═╡ cdc5dab7-9cee-4c67-9c91-5c655e5f74c7
cutoffs_fluid

# ╔═╡ 2d5dbfc0-a36a-4f15-8cf7-818295e69574
begin
	sdp_fluid = SparseDynamicsProblem(times_fluid_train, traj_fluid_train, 1, var_names = ["x", "y"])

	x_fluid, y_fluid = sdp_fluid.var_functions
	vx_fluid = LibFun((t, u) -> vx(x_fluid(t, u), y_fluid(t, u), t), "vx")
	vy_fluid = LibFun((t, u) -> vy(x_fluid(t, u), y_fluid(t, u), t), "vy") 
	lib_funs_fluid = [x_fluid, y_fluid, vx_fluid, vy_fluid]
	
	polys_fluid = polynomials(lib_funs_fluid, 2)
	add_library_function!(sdp_fluid, polys_fluid)

	options_fluid = STRidgeOptions(lambda_sparse = 0.03, lambda_ridge = 0.1, max_iters = 10)
	sdr_fluid = STRidge(sdp_fluid, options = options_fluid)

	sol_fluid_sindy = ODEProblem(sdr_fluid.odefun, ics_fluid, tspan_fluid_test) |> x -> solve(x, Tsit5())
	traj_fluid_sindy = [sol_fluid_sindy(t)[i] for t in times_fluid_test, i = 1:length(ics_fluid)]
	nothing
end

# ╔═╡ a83f6ace-ab88-461e-b95c-0d7f639c6e23
sdr_fluid.names

# ╔═╡ 39aeb386-87e1-4879-8a07-3f63df733b3f
let
	fig = Figure(size = (1000, 400))
	ax = Axis(fig[1, 1], aspect = AxisAspect(2), xlabel = "time [days]", ylabel = "Position [km]")
	labels = [L"x", L"y"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	colors = [:blue, :red]
	for i = 1:size(traj_fluid_test, 2)
	  lines!(ax, times_fluid_test, traj_fluid_test[:,i], label = labels[i], color = colors[i])
	end
	
	for i = 1:size(traj_fluid_sindy, 2)
	  lines!(ax, times_fluid_test, traj_fluid_sindy[:,i], linestyle = :dash, label = labels_sindy[i], color = colors[i])
	end

	Legend(fig[2,1], ax, orientation = :horizontal)

	###

	ax = Axis(fig[1, 2], aspect = AxisAspect(2), xlabel = L"x \,\, \text{[km]}", ylabel = L"y \,\, \text{[km]}")
	labels = [L"xy(t)"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	lines!(ax, traj_fluid_test[:,1], traj_fluid_test[:,2], label = "Fluid", color = :black)

	lines!(ax, traj_fluid_sindy[:,1], traj_fluid_sindy[:,2], linestyle = :dash, label = "Fluid [SINDy]", color = :red)
	
	Legend(fig[2,2], ax, orientation = :horizontal)


	Label(fig[:, 3], "FLUID LEARNED \n dxdt = $(sdr_fluid.names[1]) \n dydt = $(sdr_fluid.names[2])", fontsize = 20)

	fig
end

# ╔═╡ 1343467c-5130-415a-92e7-97d433afae7d
begin
	tspan_slow_test = (0.0, 120.0)
	n_times_slow = 1000 	
	tspan_slow_train = (0.0, 90.0)

	###

	sol_slow = ODEProblem(f_slow!, ics_slow, tspan_slow_test) |> x -> solve(x, Tsit5())
	
	times_slow_train = range(tspan_slow_train[1], tspan_slow_train[2], length = n_times_slow) |> collect
	traj_slow_train = [sol_slow(t)[i] for t in times_slow_train, i = 1:length(ics_slow)]
	
	times_slow_test = range(tspan_slow_test[1], tspan_slow_test[2], length = floor(Int64, (tspan_slow_test[2]/tspan_slow_train[2])*n_times_slow)) |> collect
	traj_slow_test = [sol_slow(t)[i] for t in times_slow_test, i = 1:length(ics_slow)]
	
	nothing
end

# ╔═╡ d6973577-e873-44cd-8872-d829851022a8
let
	global cutoffs_slow = []
	global sdrs_slow = []
	n_terms_poss = []
	for lambda_sparse in 10 .^ range(-3, 1, length = 100)
	  sdp_slow = SparseDynamicsProblem(times_slow_train, traj_slow_train, 1, var_names = ["x", "y"])
	
	  x_slow, y_slow = sdp_slow.var_functions
	  vx_slow = LibFun((t, u) -> vx(x_slow(t, u), y_slow(t, u), t), "vx")
	  vy_slow = LibFun((t, u) -> vy(x_slow(t, u), y_slow(t, u), t), "vy") 
	  lib_funs_slow = [x_slow, y_slow, vx_slow, vy_slow]
		
	  polys_slow = polynomials(lib_funs_slow, 2)
	  add_library_function!(sdp_slow, polys_slow)
	
      options_slow = STRidgeOptions(lambda_sparse = lambda_sparse, lambda_ridge = 0.1, max_iters = 10)
	  sdr_slow = STRidge(sdp_slow, options = options_slow)
		
	  rmse = sdr_slow.rmse |> x -> sum(x)/length(x)
	  n_terms = count(iszero, sdr_slow.coeffs)
	
	  if !(n_terms in n_terms_poss)
	    push!(cutoffs_slow, [lambda_sparse, n_terms, rmse])
	    push!(sdrs_slow, sdr_slow)
	    push!(n_terms_poss, n_terms)
	  else
	    cutoffs_slow[end] = [lambda_sparse, n_terms, rmse]
	    sdrs_slow[end] = sdr_slow
	  end
	end
	cutoffs_slow = stack(cutoffs_slow, dims = 1)
	
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = L"\log10(\lambda)", ylabel = "RMSE")
	scatter!(ax, log10.(cutoffs_slow[:,1]), cutoffs_slow[:,3], color = :black, markersize = 20)
	fig
end

# ╔═╡ 49e46fcf-8c00-45bd-9b12-8751b56b6cc6
sdrs_slow

# ╔═╡ 01962205-cb4f-4abd-a0cb-f1e0122efd1f
cutoffs_slow

# ╔═╡ afe1e2e9-f5ae-4de4-81d0-fa304efa27c5
begin
	sdp_slow = SparseDynamicsProblem(times_slow_train, traj_slow_train, 1, var_names = ["x", "y"])

	x_slow, y_slow = sdp_slow.var_functions
	vx_slow = LibFun((t, u) -> vx(x_slow(t, u), y_slow(t, u), t), "vx")
	vy_slow = LibFun((t, u) -> vy(x_slow(t, u), y_slow(t, u), t), "vy") 
	lib_funs_slow = [x_slow, y_slow, vx_slow, vy_slow]
	
	polys_slow = polynomials(lib_funs_slow, 2)
	add_library_function!(sdp_slow, polys_slow)

	options_slow = STRidgeOptions(lambda_sparse = 0.08, lambda_ridge = 0.1, max_iters = 10)
	sdr_slow = STRidge(sdp_slow, options = options_slow)

	sol_slow_sindy = ODEProblem(sdr_slow.odefun, ics_slow, tspan_slow_test) |> x -> solve(x, Tsit5())
	traj_slow_sindy = [sol_slow_sindy(t)[i] for t in times_slow_test, i = 1:length(ics_slow)]
	nothing
end

# ╔═╡ d34e7417-64be-4016-8da3-f6bd23c0acc8
sdr_slow.names

# ╔═╡ a1b9b3d2-8215-43fa-9f38-5093aca361a0
let
	fig = Figure(size = (1000, 400))
	ax = Axis(fig[1, 1], aspect = AxisAspect(2), xlabel = "time [days]", ylabel = "Position [km]")
	labels = [L"x", L"y"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	colors = [:blue, :red]
	for i = 1:size(traj_slow_test, 2)
	  lines!(ax, times_slow_test, traj_slow_test[:,i], label = labels[i], color = colors[i])
	end
	
	for i = 1:size(traj_slow_sindy, 2)
	  lines!(ax, times_slow_test, traj_slow_sindy[:,i], linestyle = :dash, label = labels_sindy[i], color = colors[i])
	end

	Legend(fig[2,1], ax, orientation = :horizontal)

	###

	ax = Axis(fig[1, 2], aspect = AxisAspect(2), xlabel = L"x \,\, \text{[km]}", ylabel = L"y \,\, \text{[km]}")
	labels = [L"xy(t)"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	lines!(ax, traj_slow_test[:,1], traj_slow_test[:,2], label = "Fluid", color = :black)

	lines!(ax, traj_slow_sindy[:,1], traj_slow_sindy[:,2], linestyle = :dash, label = "Fluid [SINDy]", color = :red)
	
	Legend(fig[2,2], ax, orientation = :horizontal)


	Label(fig[:, 3], "SLOW LEARNED \n dxdt = $(sdr_slow.names[1]) \n dydt = $(sdr_slow.names[2])", fontsize = 20)

	fig
end

# ╔═╡ 200dfd1a-0af8-4215-afae-4cf70d67813d
begin
	tspan_full_test = (0.0, 120.0)
	n_times_full = 1000 	
	tspan_full_train = (0.0, 90.0)

	###

	sol_full = ODEProblem(f_full!, [ics_full ;; ics_full_v], tspan_full_test) |> x -> solve(x, Tsit5())
	
	times_full_train = range(tspan_full_train[1], tspan_full_train[2], length = n_times_full) |> collect
	traj_full_train = [sol_full(t)[i] for t in times_full_train, i = 1:length(ics_full)]
	
	times_full_test = range(tspan_full_test[1], tspan_full_test[2], length = floor(Int64, (tspan_full_test[2]/tspan_full_train[2])*n_times_full)) |> collect
	traj_full_test = [sol_full(t)[i] for t in times_full_test, i = 1:length(ics_full)]
	
	nothing
end

# ╔═╡ b1c5a547-a08f-41eb-b60a-e181435279fe
let
	global cutoffs_full = []
	global sdrs_full = []
	n_terms_poss = []
	for lambda_sparse in 10 .^ range(-3, 1, length = 100)
	  sdp_full = SparseDynamicsProblem(times_full_train, traj_full_train, 2, var_names = ["x", "y"])
	
	  x_full, y_full = sdp_full.var_functions
	  vx_full = LibFun((t, u) -> vx(x_full(t, u), y_full(t, u), t), "vx")
	  vy_full = LibFun((t, u) -> vy(x_full(t, u), y_full(t, u), t), "vy") 
	  lib_funs_full = [x_full, y_full, vx_full, vy_full]
		
	  polys_full = polynomials(lib_funs_full, 2)
	  add_library_function!(sdp_full, polys_full)
	
      options_full = STRidgeOptions(lambda_sparse = lambda_sparse, lambda_ridge = 0.1, max_iters = 10)
	  sdr_full = STRidge(sdp_full, options = options_full)
		
	  rmse = sdr_full.rmse |> x -> sum(x)/length(x)
	  n_terms = count(iszero, sdr_full.coeffs)
	
	  if !(n_terms in n_terms_poss)
	    push!(cutoffs_full, [lambda_sparse, n_terms, rmse])
	    push!(sdrs_full, sdr_full)
	    push!(n_terms_poss, n_terms)
	  else
	    cutoffs_full[end] = [lambda_sparse, n_terms, rmse]
	    sdrs_full[end] = sdr_full
	  end
	end
	cutoffs_full = stack(cutoffs_full, dims = 1)
	
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = L"\log10(\lambda)", ylabel = "RMSE")
	scatter!(ax, log10.(cutoffs_full[:,1]), cutoffs_full[:,3], color = :black, markersize = 20)
	fig
end

# ╔═╡ afa619a2-1383-4ceb-a8a7-7beab5ecaf17
cutoffs_full

# ╔═╡ d4bf9af8-abef-40cd-9319-1862c69b886f
begin
	sdp_full = SparseDynamicsProblem(times_full_train, traj_full_train, 2, var_names = ["x", "y"])

	x_full, y_full, dx_full, dy_full = sdp_full.var_functions
	vx_full = LibFun((t, u) -> vx(x_full(t, u), y_full(t, u), t), "vx")
	vy_full = LibFun((t, u) -> vy(x_full(t, u), y_full(t, u), t), "vy") 
	lib_funs_full = [x_full, y_full, dx_full, dy_full, vx_full, vy_full]
	
	polys_full = polynomials(lib_funs_full, 1)
	add_library_function!(sdp_full, polys_full)

	options_full = STRidgeOptions(lambda_sparse = 1, lambda_ridge = 0.1, max_iters = 10)
	sdr_full = STRidge(sdp_full, options = options_full)

	sol_full_sindy = ODEProblem(sdr_full.odefun, [ics_full ; ics_full_v], tspan_full_test) |> solve
	traj_full_sindy = [sol_full_sindy(t)[i] for t in times_full_test, i = 1:length(ics_full)]
	nothing
end

# ╔═╡ fb79f103-ce0c-4bb9-b736-b75f8e823198
sdp_full

# ╔═╡ add2738b-5ab4-47ae-b977-a23709186423
sdr_full.names

# ╔═╡ 2076ae39-4f25-423f-bde1-c6e2fad0368e
let
	fig = Figure(size = (1000, 400))
	ax = Axis(fig[1, 1], aspect = AxisAspect(2), xlabel = "time [days]", ylabel = "Position [km]")
	labels = [L"x", L"y"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	colors = [:blue, :red]
	for i = 1:size(traj_full_test, 2)
	  lines!(ax, times_full_test, traj_full_test[:,i], label = labels[i], color = colors[i])
	end
	
	for i = 1:size(traj_full_sindy, 2)
	  lines!(ax, times_full_test, traj_full_sindy[:,i], linestyle = :dash, label = labels_sindy[i], color = colors[i])
	end

	Legend(fig[2,1], ax, orientation = :horizontal)

	###

	ax = Axis(fig[1, 2], aspect = AxisAspect(2), xlabel = L"x \,\, \text{[km]}", ylabel = L"y \,\, \text{[km]}")
	labels = [L"xy(t)"]
	labels_sindy = [L"x_\text{SINDy}", L"y_\text{SINDy}", L"z_\text{SINDy}"]
	lines!(ax, traj_full_test[:,1], traj_full_test[:,2], label = "Fluid", color = :black)

	lines!(ax, traj_full_sindy[:,1], traj_full_sindy[:,2], linestyle = :dash, label = "Fluid [SINDy]", color = :red)
	
	Legend(fig[2,2], ax, orientation = :horizontal)


	Label(fig[3, :], "FULL LEARNED \n dx^2dt^2 = $(sdr_full.names[1]) \n dy^2dt^2 = $(sdr_full.names[2])", fontsize = 20)

	fig
end

# ╔═╡ f9f6be75-3bd7-427e-b0cf-a49097f9467e
function f_1!(du, u, p, t)
	x, y, dxdt, dydt = u
	du[1] = dxdt
	du[2] = dydt
	du[3] = -7.68*dxdt + 7.68*dydt + 7.68*vx(x, y, t) - 6.92*vy(x, y, t)
	du[4] = -7.68*dxdt - 7.68*dydt + 6.92*vx(x, y, t) + 7.68*vy(x, y, t)
	return nothing
end

# ╔═╡ 09edf393-a054-4a62-a30e-7adda152f63a
function f_2!(du, u, p, t)
	x, y, dxdt, dydt = u
	du[1] = dxdt
	du[2] = dydt
	du[3] = -5.17*dxdt + 3.82*dydt + 5.11*vx(x, y, t) - 3.38*vy(x, y, t)
	du[4] = -8.04*dxdt - 6.31*dydt + 7.32*vx(x, y, t) + 6.40*vy(x, y, t)
	return nothing
end

# ╔═╡ 0c814347-6687-418e-a9bb-ec4ed3cc73af
begin
	# ics_p = [ics_full ; ics_full_v]
	ics_p = [ics_full .+ 100*rand() ; ics_full_v .+ rand()]
	tmax_p = 120.0
	p1 = ODEProblem(f_1!, ics_p, (0.0, tmax_p)) |> solve
	p2 = ODEProblem(f_2!, ics_p, (0.0, tmax_p)) |> solve
	nothing
end

# ╔═╡ f15eab0d-6610-4c11-9e1b-55cb58d81e76
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	lines!(ax, range(0, tmax_p, length = 100), t -> p1(t)[1], color = :red)
	lines!(ax, range(0, tmax_p, length = 100), t -> p1(t)[2], color = :blue)
	lines!(ax, range(0, tmax_p, length = 100), t -> p2(t)[1], color = :red, linestyle = :dash)
	lines!(ax, range(0, tmax_p, length = 100), t -> p2(t)[2], color = :blue, linestyle = :dash)
	fig
end

# ╔═╡ 40bf79f6-585e-4d2a-a34b-58ef397a7053
function sph2xy(lon::Real, lat::Real, eqr::EquirectangularReference)
    @assert -180.0 <= lon <= 180.0 "The longitude must be between -180 degrees and 180 degrees."
    @assert -90 <= lat <= 90 "The latitude must be between -90 degrees and 90 degrees."

    lon0, lat0, R = (eqr.lon0, eqr.lat0, eqr.R)
    deg2rad = π/180

    x = R*(lon - lon0)*deg2rad*cos(lat0*deg2rad)
    y = R*(lat - lat0)*deg2rad

    return [x, y]
end

# ╔═╡ 901d8290-c137-4a39-9390-cc422cf70f76
function sph2xy(lon_range::AbstractRange, lat_range::AbstractRange, eqr::EquirectangularReference)
    # uses the fact that the translation between eqr and spherical is linear
    lonmin, latmin = sph2xy(first(lon_range), first(lat_range), eqr)
    lonmax, latmax = sph2xy(last(lon_range), last(lat_range), eqr)

    return  (
            range(start = lonmin, length = length(lon_range), stop = lonmax), 
            range(start = latmin, length = length(lat_range), stop = latmax)
            )
end

# ╔═╡ 0fbd8b04-27cb-42be-9765-a6c746ba987a
function sph2xy(lon_lat::Matrix{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert size(lon_lat, 2) == 2 "lon_lat should be an `N x 2` matrix"
    xy = zeros(T, size(lon_lat))
    
    for i = 1:size(lon_lat, 1)
        xy[i,:] .= sph2xy(lon_lat[i,1], lon_lat[i,2], eqr)
    end

    return xy
end

# ╔═╡ 74f49db0-ca05-4d62-a826-f655418c0a03
function sph2xy(lon_lat::Vector{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert iseven(length(lon_lat)) "lon_lat should be of the form `[lon1, lat1, lon2, lat2 ... lon3, lat3]`."

    xy = zeros(T, length(lon_lat))
    
    for i = 1:2:length(lon_lat)
        xy[i:i+1] .= sph2xy(lon_lat[i], lon_lat[i + 1], eqr)
    end

    return xy
end

# ╔═╡ eb1c7b35-89e4-4b2b-a069-15d33e8ca22b
function xy2sph(x::Real, y::Real, eqr::EquirectangularReference)
    lon0, lat0, R = (eqr.lon0, eqr.lat0, eqr.R)
    deg2rad = π/180
    rad2deg = 1/deg2rad

    lon = lon0 + rad2deg*x/(R*cos(lat0*deg2rad))
    lat = lat0 + rad2deg*y/R 

    return [lon, lat]
end

# ╔═╡ cca894c8-f72d-4a2c-9346-cf8830907096
function xy2sph(xy::Vector{<:Vector{T}}, eqr::EquirectangularReference) where {T<:Real}
    lonlat = zeros(T, length(xy), 2) 
    
    for i = 1:length(xy)
        lonlat[i,:] = xy2sph(xy[i][1], xy[i][2], eqr)
    end

    return lonlat
end

# ╔═╡ c053eceb-9ea5-4193-9fbd-dfcdf67f30e2
function xy2sph(xy::Matrix{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert size(xy, 2) == 2 "xy should be an `N x 2` matrix"
    lonlat = zeros(T, size(xy))
    
    for i = 1:size(xy, 1)
        lonlat[i,:] = xy2sph(xy[i,1], xy[i,2], eqr)
    end

    return lonlat
end

# ╔═╡ 7be78e8d-7901-499a-9591-0066fb329f07
function xy2sph(x_range::AbstractRange, y_range::AbstractRange, eqr::EquirectangularReference)
    # uses the fact that the translation between eqr and spherical is linear
    xmin, ymin = xy2sph(first(x_range), first(y_range), eqr)
    xmax, ymax = xy2sph(last(x_range), last(y_range), eqr)

    return  (
            range(start = xmin, length = length(x_range), stop = xmax), 
            range(start = ymin, length = length(y_range), stop = ymax)
            )
end

# ╔═╡ bf39df89-529e-4618-8f68-a5d546e0600e
function xy2sph(xy::Vector{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert iseven(length(xy)) "xy should be of the form `[x1, y1, x2, y2 ... x3, y3]`."

    lon_lat = zeros(T, length(xy))
    
    for i = 1:2:length(xy)
        lon_lat[i:i+1] .= xy2sph(xy[i], xy[i + 1], eqr)
    end

    return lon_lat
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BSplineKit = "093aae92-e908-43d7-9660-e50ee39d5a0a"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SargassumColors = "d2143393-64d1-4702-ac44-3f87fb4b2e5f"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BSplineKit = "~0.17.2"
CairoMakie = "~0.11.11"
Combinatorics = "~1.0.2"
DifferentialEquations = "~7.13.0"
Interpolations = "~0.15.1"
Latexify = "~0.16.3"
MAT = "~0.10.6"
MultivariateStats = "~0.10.2"
SargassumColors = "~0.2.2"
StatsBase = "~0.34.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "807dee2aa451372c8dab00b9cbfa0bec11eb5ac6"

[[deps.ADTypes]]
git-tree-sha1 = "7a6b285f217ba92b5b474b783b4c2e8cf8218aaa"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.5.3"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5c9b74c973181571deb6442d41e5c902e6b9f38e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.12.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "ce2ca959f932f5dad70697dd93133d1167cf1e4e"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "014bc22d6c400a7703c0f5dc1fdc302440cf88be"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.4"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BSplineKit]]
deps = ["ArrayLayouts", "BandedMatrices", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "PrecompileTools", "Random", "Reexport", "SparseArrays", "Static", "StaticArrays", "StaticArraysCore"]
git-tree-sha1 = "79dba2b0d60f225f4660075cb0e9b4da9960042a"
uuid = "093aae92-e908-43d7-9660-e50ee39d5a0a"
version = "0.17.6"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "71f605effb24081b09cae943ba39ef9ca90c04f4"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.7.2"
weakdeps = ["SparseArrays"]

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BoundaryValueDiffEq]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "BandedMatrices", "ConcreteStructs", "DiffEqBase", "FastAlmostBandedMatrices", "FastClosures", "ForwardDiff", "LinearAlgebra", "LinearSolve", "Logging", "NonlinearSolve", "OrdinaryDiffEq", "PreallocationTools", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "Setfield", "SparseArrays", "SparseDiffTools"]
git-tree-sha1 = "4e746d02f1d7ef513c1441ee58f3b20f5d10ad03"
uuid = "764a87c0-6b3e-53db-9096-fe964310641d"
version = "5.9.0"

    [deps.BoundaryValueDiffEq.extensions]
    BoundaryValueDiffEqODEInterfaceExt = "ODEInterface"

    [deps.BoundaryValueDiffEq.weakdeps]
    ODEInterface = "54ca160b-1b9f-5127-a996-1867f4bc2a2c"

[[deps.BufferedStreams]]
git-tree-sha1 = "4ae47f9a4b1dc19897d3743ff13685925c5202ec"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "d69c7593fe9d7d617973adcbe4762028c6899b2c"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.11.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "b8fe8546d52ca154ac556809e10c75e6e7430ac8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.5"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "078c716cbb032242df18b960e8b1fec6b1b0b9f9"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.0.5"

[[deps.DelayDiffEq]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "LinearAlgebra", "Logging", "OrdinaryDiffEq", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SimpleUnPack"]
git-tree-sha1 = "5959ae76ebd198f70e9af81153644543da0cfaf2"
uuid = "bcd4f6db-9728-5f36-b5f7-82caef46ccdb"
version = "5.47.3"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PreallocationTools", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Static", "StaticArraysCore", "Statistics", "Tricks", "TruncatedStacktraces"]
git-tree-sha1 = "d1e8a4642e28b0945bde6e2e1ac569b9e0abd728"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.151.5"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDistributionsExt = "Distributions"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "Functors", "LinearAlgebra", "Markdown", "NonlinearSolve", "Parameters", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "c959cfd2657d16beada157a74d52269e8556500e"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "3.6.2"
weakdeps = ["OrdinaryDiffEq", "Sundials"]

[[deps.DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "GPUArraysCore", "LinearAlgebra", "Markdown", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "Requires", "ResettableStacks", "SciMLBase", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "65cbbe1450ced323b4b17228ccd96349d96795a7"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.21.0"

    [deps.DiffEqNoiseProcess.extensions]
    DiffEqNoiseProcessReverseDiffExt = "ReverseDiff"

    [deps.DiffEqNoiseProcess.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentialEquations]]
deps = ["BoundaryValueDiffEq", "DelayDiffEq", "DiffEqBase", "DiffEqCallbacks", "DiffEqNoiseProcess", "JumpProcesses", "LinearAlgebra", "LinearSolve", "NonlinearSolve", "OrdinaryDiffEq", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SteadyStateDiffEq", "StochasticDiffEq", "Sundials"]
git-tree-sha1 = "81042254a307980b8ab5b67033aca26c2e157ebb"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "7.13.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "FillArrays", "LinearAlgebra", "PackageExtensionCompat", "SparseArrays", "SparseMatrixColorings"]
git-tree-sha1 = "695217e97ee1ce0248f4a56c14af88ba33c585fd"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.5.7"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = "Enzyme"
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = "ForwardDiff"
    DifferentiationInterfacePolyesterForwardDiffExt = "PolyesterForwardDiff"
    DifferentiationInterfaceReverseDiffExt = "ReverseDiff"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTapirExt = "Tapir"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tapir = "07d77754-e150-4737-8c94-cd238a1fb45b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EnzymeCore]]
git-tree-sha1 = "d445df66dd8761a4c27df950db89c6a3a0629fe7"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.7.7"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.ExponentialUtilities]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "GenericSchur", "LinearAlgebra", "PrecompileTools", "Printf", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "8e18940a5ba7f4ddb41fe2b79b6acaac50880a86"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.26.1"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "94997910aca72897524d2237c41eb852153b0f65"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.3"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastAlmostBandedMatrices]]
deps = ["ArrayInterface", "ArrayLayouts", "BandedMatrices", "ConcreteStructs", "LazyArrays", "LinearAlgebra", "MatrixFactorizations", "PrecompileTools", "Reexport"]
git-tree-sha1 = "a92b5820ea38da3b50b626cc55eba2b074bb0366"
uuid = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
version = "0.1.3"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "bd19de6fe8a3b18888f35e79832f97544684caa7"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.4"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FastLapackInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cbf5edddb61a43669710cbc2241bc08b36d9e660"
uuid = "29a986be-02c6-4525-aec4-84b980013641"
version = "2.0.4"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8a66c07630d6428eaab3506a0eabfcf4a9edea05"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.11"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "af49a0851f8113fcfae2ef5027c6d49d0acec39b"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.4"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "59107c179a586f0fe667024c5eb7033e81333271"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.2"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "9fff8990361d5127b770e3454488360443019bb3"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.5"

[[deps.GeoInterfaceMakie]]
deps = ["GeoInterface", "GeometryBasics", "MakieCore"]
git-tree-sha1 = "3f87fd8414194dd25ea5d0371c3950985e3c8d86"
uuid = "0edc0954-3250-4c18-859d-ec71c1660c08"
version = "0.1.8"

[[deps.GeoInterfaceRecipes]]
deps = ["GeoInterface", "RecipesBase"]
git-tree-sha1 = "fb1156076f24f1dfee45b3feadb31d05730a49ac"
uuid = "0329782f-3d07-4b52-b9f6-d3137cf03c7a"
version = "1.0.2"

[[deps.GeoJSON]]
deps = ["Extents", "GeoFormatTypes", "GeoInterface", "GeoInterfaceMakie", "GeoInterfaceRecipes", "JSON3", "StructTypes", "Tables"]
git-tree-sha1 = "e2ae0c6d4f6b8c49eccc261fef29c290998e44a5"
uuid = "61d90e0f-e114-555e-ac52-39dfb47a3ef9"
version = "0.8.1"
weakdeps = ["Makie"]

    [deps.GeoJSON.extensions]
    GeoJSONMakieExt = "Makie"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ebd18c326fa6cee1efb7da9a3b45cf69da2ed4d9"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.11.2"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "6f93a83ca11346771a93bbde2bdad2f65b61498f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1d334207121865ac8c1c97eb7f42d0339e4635bf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.0+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14eb2b542e748570b56446f4c50fbfb2306ebc45"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "433b0bb201cd76cb087b017e49244f10394ebe9c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.14"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "e7cbed5032c4c397a6ac23d1493f3289e01231c4"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.14"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "eb3edce0ed4fa32f75a0a11217433c31d56bd48b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.0"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.JumpProcesses]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "Markdown", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SciMLBase", "StaticArrays", "SymbolicIndexingInterface", "UnPack"]
git-tree-sha1 = "ed08d89318be7d625613f3c435d1f6678fba4850"
uuid = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
version = "9.11.1"
weakdeps = ["FastBroadcast"]

    [deps.JumpProcesses.extensions]
    JumpProcessFastBroadcastExt = "FastBroadcast"

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "07649c499349dad9f08dde4243a4c597064663e9"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.6.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "267dad6b4b7b5d529c76d40ff48d33f7e94cb834"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "5b0d630f3020b82c0775a51d05895852f8506f50"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.4"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "46dd13736e33cc3bfc610f62b6c7f84b9c95539a"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.1.8"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LevyArea]]
deps = ["LinearAlgebra", "Random", "SpecialFunctions"]
git-tree-sha1 = "56513a09b8e0ae6485f34401ea9e2f31357958ec"
uuid = "2d8b4e74-eb68-11e8-0fb9-d5eb67b50637"
version = "1.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "b2e2dba60642e07c062eb3143770d7e234316772"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "2.30.2"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveEnzymeExt = ["Enzyme", "EnzymeCore"]
    LinearSolveFastAlmostBandedMatricesExt = ["FastAlmostBandedMatrices"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolveRecursiveArrayToolsExt = "RecursiveArrayTools"

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "4099bb6809ac109bfc17d521dad33763bcf026b7"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.1+1"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8c35d5420193841b2f367e658540e8d9e0601ed0"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.4.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "4d49c9ee830eec99d3e8de2425ff433ece7cc1bc"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.10"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "07c98fdf57c9b45b987cf250c4bdc7200fa39eb2"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "3.0.0"
weakdeps = ["BandedMatrices"]

    [deps.MatrixFactorizations.extensions]
    MatrixFactorizationsBandedMatricesExt = "BandedMatrices"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "1b9e613f2ca3b6cdcbfe36381e17ca2b66d4b3a1"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NaturalEarth]]
deps = ["Downloads", "GeoJSON", "Pkg", "Scratch"]
git-tree-sha1 = "3f75210ac08fe4496a55f9694b95859c40b8eaea"
uuid = "436b0209-26ab-4e65-94a9-6526d86fea76"
version = "0.1.0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "LazyArrays", "LineSearches", "LinearAlgebra", "LinearSolve", "MaybeInplace", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SparseArrays", "SparseDiffTools", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "3adb1e5945b5a6b1eaee754077f25ccc402edd7f"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "3.13.1"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveBandedMatricesExt = "BandedMatrices"
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = "NLsolve"
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSymbolicsExt = "Symbolics"
    NonlinearSolveZygoteExt = "Zygote"

    [deps.NonlinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a028ee3cb5641cccc4c24e90c36b0a4f7707bdf5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.OrdinaryDiffEq]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "ExponentialUtilities", "FastBroadcast", "FastClosures", "FillArrays", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "IfElse", "InteractiveUtils", "LineSearches", "LinearAlgebra", "LinearSolve", "Logging", "MacroTools", "MuladdMacro", "NonlinearSolve", "Polyester", "PreallocationTools", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "SimpleNonlinearSolve", "SimpleUnPack", "SparseArrays", "SparseDiffTools", "Static", "StaticArrayInterface", "StaticArrays", "TruncatedStacktraces"]
git-tree-sha1 = "6ef13f8b23af28ee2d98226653d8382ab79287ea"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.85.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cb5a2ab6763464ae0f19c86c56c63d4a2b0f5bda"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PoissonRandom]]
deps = ["Random"]
git-tree-sha1 = "a0f1159c33f846aa77c3f30ebbc69795e5327152"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.4"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "9ff799e8fb8ed6717710feee3be3bc20645daa97"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.15"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "406c29a7f46706d379a3bce45671b4e3a39ddfbc"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.22"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "80686d28ecb3ee7fb3ac5371cacaa0d673eb0d4a"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.1"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "b450d967a770fb13d0e26358f58375e20361cf9c"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.26.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "6db1a75507051bc18bfa131fbc7c3f169cc4b2f6"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.23"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "2803cab51702db743f3fda07dd1745aadfbf43bd"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.5.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SargassumColors]]
deps = ["ArgCheck", "ColorSchemes", "Colors", "GeoInterface", "GeoJSON", "GeometryBasics", "Makie", "NaturalEarth"]
git-tree-sha1 = "b4d51fa1730ae1a8491a863a30ee7aa55554d45b"
uuid = "d2143393-64d1-4702-ac44-3f87fb4b2e5f"
version = "0.2.6"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "281e82f2ae2b73262fed9e7a518711eb7feb7e59"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.42.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Setfield", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "10499f619ef6e890f3f4a38914481cc868689cd5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.8"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "cfdd1200d150df1d3c055cc72ee6850742e982d7"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.4.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "DiffResults", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "Reexport", "SciMLBase", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "58b144f34e44252b2de0acb5a9dbbb7ea5cd75d7"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "1.10.1"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"
    SimpleNonlinearSolveZygoteExt = "Zygote"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseDiffTools]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "PackageExtensionCompat", "Random", "Reexport", "SciMLOperators", "Setfield", "SparseArrays", "StaticArrayInterface", "StaticArrays", "Tricks", "UnPack", "VertexSafeGraphs"]
git-tree-sha1 = "469f51f8c4741ce944be2c0b65423b518b1405b0"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "2.19.0"

    [deps.SparseDiffTools.extensions]
    SparseDiffToolsEnzymeExt = "Enzyme"
    SparseDiffToolsPolyesterExt = "Polyester"
    SparseDiffToolsPolyesterForwardDiffExt = "PolyesterForwardDiff"
    SparseDiffToolsSymbolicsExt = "Symbolics"
    SparseDiffToolsZygoteExt = "Zygote"

    [deps.SparseDiffTools.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "eed2446b3c3dd58f6ded3168998b8b2cb3fc9229"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.3.3"

[[deps.Sparspak]]
deps = ["Libdl", "LinearAlgebra", "Logging", "OffsetArrays", "Printf", "SparseArrays", "Test"]
git-tree-sha1 = "342cf4b449c299d8d1ceaf00b7a49f4fbc7940e7"
uuid = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
version = "0.3.9"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "8963e5a083c837531298fc41599182a759a87a6d"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.SteadyStateDiffEq]]
deps = ["ConcreteStructs", "DiffEqBase", "DiffEqCallbacks", "LinearAlgebra", "Reexport", "SciMLBase"]
git-tree-sha1 = "1158cfdf0da5b0eacdfcfba7c16b174a37bdf6c7"
uuid = "9672c7b4-1e72-59bd-8a11-6ac3964bc41f"
version = "2.2.0"

[[deps.StochasticDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqNoiseProcess", "DocStringExtensions", "FiniteDiff", "ForwardDiff", "JumpProcesses", "LevyArea", "LinearAlgebra", "Logging", "MuladdMacro", "NLsolve", "OrdinaryDiffEq", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "5237f2ebdf3b6b7ad2ec635440b59a390988feeb"
uuid = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"
version = "6.66.0"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.Sundials]]
deps = ["CEnum", "DataStructures", "DiffEqBase", "Libdl", "LinearAlgebra", "Logging", "PrecompileTools", "Reexport", "SciMLBase", "SparseArrays", "Sundials_jll"]
git-tree-sha1 = "e15f5a73f0d14b9079b807a9d1dac13e4302e997"
uuid = "c3572dad-4567-51f8-b174-8c6c989267f4"
version = "4.24.0"

[[deps.Sundials_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "SuiteSparse_jll", "libblastrampoline_jll"]
git-tree-sha1 = "ba4d38faeb62de7ef47155ed321dce40a549c305"
uuid = "fb77eaff-e24c-56d4-86b1-d163f2edb164"
version = "5.2.2+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "9c490ee01823dc443da25bf9225827e3cdd2d7e9"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.26"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "bc7fd5c91041f44636b2c134041f7e5263ce58ae"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.TranscodingStreams]]
git-tree-sha1 = "60df3f8126263c0d6b357b9a1017bb94f53e3582"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.0"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "be986ad9dac14888ba338c2554dcfec6939e1393"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.2.1"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "e7f5b81c65eb858bed630fe006837b935518aca5"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.70"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "d9717ce3518dc68a99e6b96300813760d887a01d"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.1+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─5725e2ce-40ec-42ee-b521-2c54922bad7f
# ╠═8976a397-bbaa-4439-b19a-09dc87258f45
# ╟─5fad43f1-80e3-43ae-99bf-daf288b27e4f
# ╠═ec34c8de-62fc-4f0d-b05d-43423d09d823
# ╠═877ef8d8-2499-4692-a26a-b345efbd597a
# ╠═e6db750f-73bc-4f4a-b6d2-ff27a575d694
# ╠═831a346a-3200-497f-87c3-f6bca5d2e146
# ╠═aa630bbe-1447-4a89-b32f-86911d35661c
# ╠═4621b7dd-530e-4488-8952-8581195ec976
# ╠═0eca8a7c-ec2a-4b97-a8cf-65282bc5b7e0
# ╠═ee9f84a0-a949-45b4-81d4-e8ac73236420
# ╠═74742e45-c5d5-45a1-b161-f63ab82bd0d7
# ╠═7532911c-da81-4ab9-92da-affa677d6e27
# ╠═1b3ed7d7-bb7c-4faf-ab89-5c65f816578b
# ╟─7388ca53-d7aa-4e95-a830-c99381550413
# ╠═0d4d2c16-0669-451b-99ee-8ea93c172f87
# ╠═5b5eb1dd-ab3e-4f6e-8189-c123c6505252
# ╠═22a674b3-5fa3-4cfe-b854-d3fef37e6f02
# ╠═634cbbb0-a1eb-4b4c-ad19-1bc4e56b5ae8
# ╠═35a4d5d1-a490-4853-a141-48bbc8a194da
# ╟─be9c7a59-9a2a-41d8-8f63-d4919b6cee4e
# ╟─31f79ec2-90ec-4381-b112-f682f01e66c9
# ╠═5d04a43f-98d8-475c-97a5-17c0db2a259c
# ╠═3bd280c2-e6c3-411e-a4e8-5969afce0dd7
# ╟─77c2ea65-cd7b-45ce-b168-7f4e5280628b
# ╠═0e51b25d-55d0-4c0d-b2fd-3e951329266f
# ╠═bb5174ab-523e-4363-ba47-38d9c47f0de4
# ╠═cd6228f3-8b1c-40e1-bda4-b62de5d9b483
# ╠═cdc5dab7-9cee-4c67-9c91-5c655e5f74c7
# ╠═2d5dbfc0-a36a-4f15-8cf7-818295e69574
# ╠═a83f6ace-ab88-461e-b95c-0d7f639c6e23
# ╠═39aeb386-87e1-4879-8a07-3f63df733b3f
# ╟─31b759e3-60fc-4c9a-aa7c-053906125d0b
# ╠═7ffdd06a-1462-45a7-8b42-836db4a93aad
# ╠═1343467c-5130-415a-92e7-97d433afae7d
# ╠═d6973577-e873-44cd-8872-d829851022a8
# ╠═49e46fcf-8c00-45bd-9b12-8751b56b6cc6
# ╠═01962205-cb4f-4abd-a0cb-f1e0122efd1f
# ╠═afe1e2e9-f5ae-4de4-81d0-fa304efa27c5
# ╠═d34e7417-64be-4016-8da3-f6bd23c0acc8
# ╠═a1b9b3d2-8215-43fa-9f38-5093aca361a0
# ╟─8d8e2bdc-c35e-4332-884a-34bc49707ffa
# ╠═49483d2c-a284-4643-8000-ef5d664b4e18
# ╠═200dfd1a-0af8-4215-afae-4cf70d67813d
# ╠═b1c5a547-a08f-41eb-b60a-e181435279fe
# ╠═afa619a2-1383-4ceb-a8a7-7beab5ecaf17
# ╠═d4bf9af8-abef-40cd-9319-1862c69b886f
# ╠═fb79f103-ce0c-4bb9-b736-b75f8e823198
# ╠═add2738b-5ab4-47ae-b977-a23709186423
# ╠═2076ae39-4f25-423f-bde1-c6e2fad0368e
# ╟─58b47ca8-fa19-44f0-922a-58ac71816eb4
# ╠═f9f6be75-3bd7-427e-b0cf-a49097f9467e
# ╠═09edf393-a054-4a62-a30e-7adda152f63a
# ╠═0c814347-6687-418e-a9bb-ec4ed3cc73af
# ╠═f15eab0d-6610-4c11-9e1b-55cb58d81e76
# ╟─788f628e-c77c-49a5-ac81-a527ede9cfbd
# ╟─6ca5c925-621e-4d6f-a379-e20d4f7bc225
# ╠═840cada2-6d5f-4bad-bee7-5faba2071a08
# ╠═40d8c53e-c18d-48d5-93d2-b5afb42ab962
# ╠═40bf79f6-585e-4d2a-a34b-58ef397a7053
# ╠═901d8290-c137-4a39-9390-cc422cf70f76
# ╠═0fbd8b04-27cb-42be-9765-a6c746ba987a
# ╠═74f49db0-ca05-4d62-a826-f655418c0a03
# ╠═eb1c7b35-89e4-4b2b-a069-15d33e8ca22b
# ╠═cca894c8-f72d-4a2c-9346-cf8830907096
# ╠═c053eceb-9ea5-4193-9fbd-dfcdf67f30e2
# ╠═7be78e8d-7901-499a-9591-0066fb329f07
# ╠═bf39df89-529e-4618-8f68-a5d546e0600e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
