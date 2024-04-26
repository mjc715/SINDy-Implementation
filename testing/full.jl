include(joinpath(@__DIR__, "..", "sindy2.jl"))
include(joinpath(@__DIR__, "..", "data", "traj.jl"))


prob_vec = generate_trajectory("full")
sol_vec = solve(prob_vec)

# times = sol_vec.t
# trajectories = stack(sol_vec.u, dims = 1)
times = range(sol_vec.t[1], sol_vec.t[end], length = 100) |> collect
trajectories = stack(sol_vec.(times), dims = 1)[:,1:2]

X(t, traj) = traj[1](t)
Y(t, traj) = traj[2](t)
dX(t, traj) = traj[3](t)
dY(t, traj) = traj[4](t)

f0(t, traj) = 1.0
f1(t, traj) = X(t, traj)
f2(t, traj) = Y(t, traj)
f3(t, traj) = dX(t, traj)
f4(t, traj) = dY(t, traj)
f5(t, traj) = X(t, traj)^2
f6(t, traj) = Y(t, traj)^2
f7(t, traj) = X(t, traj)*Y(t, traj)
f10(t, traj) = vx(X(t, traj), Y(t, traj), t)
f11(t, traj) = vy(X(t, traj), Y(t, traj), t)

### ocean
library = [f1, f2, f3, f4, f5, f6, f7, f10, f11]
library_names = ["x", "y", "dxdt", "dydt", "x^2", "y^2", "xy", "v_x", "v_y"]

si = sindy(times, trajectories, library, order = 2, library_names = library_names, λ_sparse = 0.2)

# du[3] = f * (dydt - δ * vy(x, y, t)) + (vx(x, y, t) - dxdt) / τ
# du[4] = f * (δ * vx(x, y, t) - dxdt) + (vy(x, y, t) - dydt) / τ

c1 = round(f , sigdigits = 3)
c2 = round(f*δ , sigdigits = 3)
c3 = round(1/τ , sigdigits = 3)

@info md"""
THEORY

dx^2/dt^2 = -$(c3)*dxdt + $(c1) dydt + $(c3)*vx(x, y, t) - $(c2) vy(x, y, t)

dy^2/dt^2 = -$(c1)*dxdt - $(c3) dydt + $(c2) vx(x, y, t) + $(c3) vy(x, y, t) 
"""

@info md"""
SINDy

$(si[1])

$(si[2])
"""
