# Sparse representation algorithm
#####################################

f1(x) = x
f2(x) = x^2
f3(x) = x^3
f4(y) = y

library = [f1 f2 f3]
n_iterations = 10
n_vars = 1
data = [
    0 1 2 3 4 5;
    0 2 4 6 8 10
] # x,y

theta = [
    1 1 1 1 1 1;
    0 1 2 3 4 5;
    0 2 4 6 8 10;
    0 2 8 18 32 50;
    0 1 4 9 16 25
] # 1,x,y,xy,x^2,y^2
theta = permutedims(theta)
lambda = 0.25
data = permutedims(data)

Xi = vec(theta \ data)
for k in 1:n_iterations
    smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
    # println(smallinds)
    Xi[smallinds] .= 0
    biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
    col = []
    # println(biginds)
    for j in eachindex(biginds)
        if biginds[j] <= 5
            append!(col, 1)
        else
            append!(col, 2)
        end
    end
    println(col)
    helper = biginds .% 5
    println(helper)
    Xi[biginds] = theta[:, helper] \ data[:, col]
end
println(Xi)


######################################
function sparse_representation(library, data, lambda, n_iterations, n_vars)
    # theta = [library[i](x) for x in data[1, :], i = 1:length(library)]
    theta = [
        1 1 1 1 1 1;
        0 1 2 3 4 5;
        0 2 4 6 8 10;
        0 2 8 18 32 50;
        0 1 4 9 16 25
    ] # 1,x,y,xy,x^2,y^2
    theta = permutedims(theta)
    # println(theta)
    Xi = vec(theta \ data)
    # vec(theta)
    # vec(data)
    # println("before: ", Xi)

    for k in 1:n_iterations
        smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
        println(smallinds)
        Xi[smallinds] .= 0
        for ind in 1:n_vars
            biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
            println(biginds)
            Xi[biginds] = theta[:, biginds] \ data[ind, :]
        end
    end
    # println("after: ", Xi)
    return Xi
end
###################################### make theta vector of matricies w/ each matrix like eq 4a

# results = sparse_representation(library, data, lambda, n, n_vars)
# println(results)
# for var in 1:n_vars
#     for i in 1:length(results[:, var])
#         println(String("$(results[i, var]) f$(i)"))
#     end
# end

### plan
# 1. write function for 1 variable case
# 2. put ind back in to get multivariable case (vectorize)
# 3. true SINDy