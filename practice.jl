# Sparse representation algorithm
#####################################

f1(x) = x
f2(x) = x^2
f3(x) = x^3
f4(y) = y

library = [f1 f2 f3]
n = 10
n_vars = 1
data = [
    0 1 2 3 4 5;
    0 1 4 9 16 25
] # t, dxdt
lambda = 0.25

######################################
function sparse_representation(library, data, lambda, n_iterations, n_vars)
    theta = [library[i](x) for x in data[1, :], i = 1:length(library)]
    # println(theta)
    Xi = theta \ data[2, :]
    # println("before: ", Xi)

    for k in 1:n_iterations
        smallinds = findall(p -> (p < lambda && p > -lambda), Xi) #array of indicies with small coefficients
        Xi[smallinds] .= 0
        for ind in 1:n_vars
            biginds = [i for i = 1:length(Xi[ind]) if !(i in smallinds)]
            Xi[biginds] = theta[:, biginds] \ data[ind+1, :]
        end
    end
    # println("after: ", Xi)
    return Xi
end
######################################

results = sparse_representation(library, data, lambda, n, n_vars)

for var in 1:n_vars
    for i in 1:length(results[:, var])
        println(String("$(results[i, var]) f$(i)"))
    end
end

### plan
# 1. write function for 1 variable case
# 2. put ind back in to get multivariable case
# 3. true SINDy