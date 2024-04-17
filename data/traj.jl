using DifferentialEquations
using MAT
using Interpolations
using Distributions

xyt0 = matread(joinpath(@__DIR__, "sargassum-xyorigin.mat"))
lon_origin = xyt0["lon_origin"]
lat_origin = xyt0["lat_origin"]

xyt = matread(joinpath(@__DIR__, "sargassum-xyt.mat"))
x, y, t = (xyt["x"], xyt["y"], xyt["t"]) .|> vec
x = range(x[1], x[end], length = length(x))
y = range(y[1], y[end], length = length(y))
t = range(0, t[end] - t[1], length = length(t))

uv = matread(joinpath(@__DIR__, "sargassum-uv.mat"))
vx_data, vy_data = (uv["u"], uv["v"]) .|> x -> permutedims(x, (2, 1, 3))

vx = cubic_spline_interpolation((x, y, t), vx_data)
vy = cubic_spline_interpolation((x, y, t), vy_data)

###

function f_fluid!(du, u, p, t)
    x, y = u
    du[1] = vx(x, y, t)
    du[2] = vy(x, y, t)
    return nothing
end

f = 2 * 2*π * sin(lat_origin*π/180)
δ = 0.9
τ = 1/f

function f_slow!(du, u, p, t)
    x, y = u
    du[1] = vx(x, y, t) + f*τ*(1 - δ)*vy(x, y, t)
    du[2] = vy(x, y, t) - f*τ*(1 - δ)*vx(x, y, t)
    return nothing
end

function f_full!(du, u, p, t)
    x, y, dxdt, dydt = u
    du[1] = dxdt
    du[2] = dydt
    du[3] = f*(dydt - δ*vy(x,y,t)) + (vx(x,y,t) - dxdt)/τ
    du[4] = f*(δ*vx(x,y,t) - dxdt) + (vy(x,y,t) - dydt)/τ
    return nothing
end

###

"""
    generate_trajectory(type)

Generate a single trajectory initialized randomly, integrating equations of type `type`.

Returns an `ODESolution`.

### Types

- `"fluid"`: Water trajectory (x, y).
- `"slow"`: Slow manifold Maxey-Riley approximation (x, y).
- `"full"`: Full Maxey-Riley equations (x, y, vx, vy).
"""
function generate_trajectory(type::String)
    @assert type in ["fluid", "slow", "full"]

    xy0 = [rand(Uniform(extrema(x)...)), rand(Uniform(extrema(y)...))]
    t1 = rand(Uniform(0, maximum(t)/2))
    t2 = rand(Uniform(t1 + 10, maximum(t) - 1))
    tspan = (t1, t2)

    if type == "fluid"
        prob = ODEProblem(f_fluid!, xy0, tspan)
    elseif type == "slow"
        prob = ODEProblem(f_slow!, xy0, tspan)
    elseif type == "full"
        dxdy0 = [rand(), rand()]
        xy0 = [xy0 ;; dxdy0]
        prob = ODEProblem(f_full!, xy0, tspan)
    end

    try 
        solve(prob, Tsit5())
        return prob
    catch
        return generate_trajectory(type)
    end
end