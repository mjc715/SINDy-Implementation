# Sparse representation algorithm

# x,x^2,x^3
theta = [1 1 1;
    2 4 8;
    3 9 27]

dXdt = [2.1; 6.1; 11.9] # x^2 + x w/ added noise

lambda = 0.2 # sparcification parameter
n = 3
Xi = theta \ dXdt

for k in 1:10
    smallinds = findall(<(lambda), Xi) #array of indicies with small coefficients
    Xi[smallinds] .= 0
    for j in 1:n
        biginds = findall(~smallinds[:, j], smallinds)
        Xi[biginds, j] = theta[:, biginds] \ dXdt[:, ind]
    end
end

print(Xi)
