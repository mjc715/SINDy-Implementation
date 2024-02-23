# Sparse representation algorithm

# x,x^2,x^3
theta = [1 1 1;
    2 4 8;
    3 9 27]

dXdt = [2.1; 6.1; 11.9] # x^2 + x w/ added noise

lambda = 0.2 # sparcification parameter
n = 1
Xi = theta \ dXdt
println("before: ", Xi)
for k in 1:10
    biginds = []
    smallinds = findall(<(lambda), abs.(Xi)) #array of indicies with small coefficients
    Xi[smallinds] .= 0
    for ind in 1:n
        for i in 1:size(Xi)[1]
            if i ∉ smallinds
                append!(biginds, i)
            end
        end
        # biginds = findall(~smallinds[:, j], smallinds)
        Xi[biginds, ind] = theta[:, biginds] \ dXdt[:, ind]
    end
end

print("after: ", Xi)
