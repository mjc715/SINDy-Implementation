include(joinpath(@__DIR__, "..", "sindy2.jl"))

function lorenz!(du, u, p, t)
    x, y, z = u
    du[1] = 10.0 * (y - x)
    du[2] = x * (28.0 - z) - y
    du[3] = x * y - (8 / 3) * z
end

prob_vec = ODEProblem(lorenz!, [1.0, 1.0, 1.0], (0.0, 10.0))
sol_vec = solve(prob_vec)

# times = sol_vec.t
# trajectories = stack(sol_vec.u, dims = 1)
times = range(sol_vec.t[1], sol_vec.t[end], length = 1000) |> collect
trajectories = stack(sol_vec.(times), dims = 1)

X(t, traj) = traj[1](t)
Y(t, traj) = traj[2](t)
Z(t, traj) = traj[3](t)
f0(t, traj) = 1.0
f1(t, traj) = X(t, traj)
f2(t, traj) = Y(t, traj)
f3(t, traj) = Z(t, traj)
f4(t, traj) = X(t, traj)^2
f5(t, traj) = Y(t, traj)^2
f6(t, traj) = Z(t, traj)^2
f7(t, traj) = X(t, traj)*Y(t, traj)
f8(t, traj) = X(t, traj)*Z(t, traj)
f9(t, traj) = Y(t, traj)*Z(t, traj)

library = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]
library_names = ["", "x", "y", "z", "x^2", "y^2", "z^2", "xy", "xz", "yz"]

si = sindy(times, trajectories, library, order = 1, library_names = library_names, λ_sparse = 0.9)