# Sparse representation algorithm

# x,x^2,x^3
theta = [1 1 1;
    2 4 8;
    3 9 27;
    4 16 64]

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

# test
