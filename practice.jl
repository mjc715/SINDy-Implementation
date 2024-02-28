# Sparse representation algorithm

# x,x^2,x^3
theta = [
    1 1 1;
    2 4 8;
    3 9 27;
    4 16 64
]

dXdt = [2.3; 6.1; 11.7; 20] # x^2 + x w/ added noise

lambda = 0.2 # sparcification parameter
Xi = theta \ dXdt
println("before: ", Xi)

for k in 1:10
    smallinds = findall(<(lambda), abs.(Xi)) #array of indicies with small coefficients
    Xi[smallinds] .= 0
    biginds = [i for i = 1:length(Xi) if !(i in smallinds)]

    Xi[biginds] = theta[:, biginds] \ dXdt
end

print("after: ", Xi)

#####################################

f1(x) = x
f2(x) = x^2
f3(x) = x^3

library = [f1, f2, f3]

data = [1.0, 2.0, 3.0, 4.0, 5.0]

theta = [library[i](x) for x in data, i = 1:length(library)]

# library = vector of functions
function sparse_representation(library, x_values, lambda, n_iterations)
    ###
end

### plan
# 1. write function for 1 variable case
# 2. put ind back in to get multivariable case
# 3. true SINDy