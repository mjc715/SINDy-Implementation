include(joinpath(@__DIR__, "..", "sindy2.jl"))
include(joinpath(@__DIR__, "..", "data", "traj.jl"))


prob_vec = generate_trajectory("slow")
sol_vec = solve(prob_vec)

# times = sol_vec.t
# trajectories = stack(sol_vec.u, dims = 1)
times = range(sol_vec.t[1], sol_vec.t[end], length = 100) |> collect
trajectories = stack(sol_vec.(times), dims = 1)

X(t, traj) = traj[1](t)
Y(t, traj) = traj[2](t)
f0(t, traj) = 1.0
f1(t, traj) = X(t, traj)
f2(t, traj) = Y(t, traj)
f4(t, traj) = X(t, traj)^2
f5(t, traj) = Y(t, traj)^2
f7(t, traj) = X(t, traj)*Y(t, traj)
f10(t, traj) = vx(X(t, traj), Y(t, traj), t)
f11(t, traj) = vy(X(t, traj), Y(t, traj), t)

### ocean
library = [f1, f2, f4, f5, f7, f10, f11]
library_names = ["x", "y", "x^2", "y^2", "xy", "v_x", "v_y"]

si = sindy(times, trajectories, library, order = 1, library_names = library_names, λ_sparse = 0.05)

c = round(f * τ * (1 - δ), sigdigits = 3)

@info md"""
THEORY

dxdt = vx(x, y, t) + $(c) * vy(x, y, t)

dydt = vy(x, y, t) - $(c) * vx(x, y, t)
"""

@info md"""
SINDy

$(si[1])

$(si[2])
"""