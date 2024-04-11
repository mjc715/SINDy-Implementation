# Sparse representation algorithm
#####################################
f0(x) = 1
f1(x) = x
f2(x) = x^2
f3(x) = x^3
f4(y) = (sin(y))
f5(y) = (sin(y^2))
f6(y) = sin(y^3)
f7(x) = f1(x) * f4(x)
f8(x) = f1(x) * f5(x)
f9(x) = f2(x) * f4(x)
f10(x) = f2(x) * f5(x)

library = [f0, f1, f2, f4, f7]
times = [i for i = 1:10]
vars = 2
data1 = [f2(t) - f1(t) for t in times] # 1. x^2-x
data2 = [f7(t) - 3 for t in times]
data = [data1; data2] # 2. xy - 3 (x = t, y = sin(t))

function sparse_representation(library, data, times, vars)
    helper = []
    n_iterations = 10
    lambda = 0.25

    for f in library
        i = 1
        row = zeros(length(times))
        for t in times
            row[i] = f(t)
            i += 1
        end
        push!(helper, row)

    end

    theta = zeros(length(times), length(library))
    dims = size(theta)

    for c in 1:dims[2]
        for r in 1:dims[1]
            theta[r, c] = helper[c][r]
        end
    end

    data = vec(permutedims(data))
    # print(data)
    data_arrays = []

    if vars == 2
        midpoint = Integer(length(data) / 2)
        row1 = data[1:midpoint]
        row2 = data[midpoint+1:length(data)]
        data_arrays = [[row1] [row2]]
    elseif vars == 1
        data_arrays = data
    end


    for l in 1:vars
        Xi = theta \ data_arrays[l]
        # print(Xi)
        for k in 1:n_iterations
            smallinds = findall(p -> (abs(p) < abs(lambda)), Xi) #array of indicies with small coefficients
            # println(smallinds)
            Xi[smallinds] .= 0
            biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
            # println(biginds)
            Xi[biginds] = theta[:, biginds] \ data_arrays[l]
        end
        println(l, ": ", Xi)
    end
end

# theta = sparse_representation(library, data, times, vars)


########### Scalar Multivariable ####################
# times = [i for i = 0:20]
# data = [f3(t) - f7(t) + 4 for t in times]
# row_1 = [1 for t in times]
# row_x = [f1(t) for t in times]
# row_y = [f4(t) for t in times]
# row_xy = [f1(t) * f4(t) for t in times]
# row_x3 = [f3(t) for t in times]
# theta = [row_1;; row_x;; row_y;; row_xy;; row_x3]
# lambda = 0.25
# data = vec(permutedims(data)) # Vector of data, x then y
# Xi = theta \ data
# for k in 1:n_iterations
#     smallinds = findall(p -> (abs(p) < abs(lambda)), Xi) #array of indicies with small coefficients
#     # println(smallinds)
#     Xi[smallinds] .= 0
#     biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
#     # println(biginds)
#     Xi[biginds] = theta[:, biginds] \ data
#     # print(Xi)
# end
# println(Xi)
#####################################################
########### Vector Multivariable ####################
# n_iterations = 10
# times = range(1, 10, length=100)
# data2 = [f7(t) - f3(t) for t in times]
# data1 = [f7(t) + 3 for t in times]
# lentimes = length(times)
# zed = zeros(Integer(length(times) / 2))
# # times = [[i for i = 1:10]; [i for i = 1:10]]
# row_1 = [1 for t in times]
# row_x = [f1(t) for t in times]
# row_x2 = [f2(t) for t in times]
# row_x3 = [f3(t) for t in times]
# row_y = [f1(2t) for t in times]
# row_y2 = [f5(t) for t in times]
# row_y3 = [f6(t) for t in times]
# row_xy = [f1(t) * f1(2t) for t in times]

# theta = [row_1;; row_x;; row_x2;; row_x3;; row_xy]

# lambda = 0.25
# datas = [[data1] [data2]]
# for l in 1:size(datas, 2)
#     Xi = theta \ datas[l]
#     for k in 1:n_iterations
#         smallinds = findall(p -> (abs(p) < abs(lambda)), Xi) #array of indicies with small coefficients
#         # println(smallinds)
#         Xi[smallinds] .= 0
#         biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
#         # println(biginds)
#         Xi[biginds] = theta[:, biginds] \ datas[l]
#     end
#     println(l, ": ", Xi)
# end
######################################################
###################################### make theta vector of matricies w/ each matrix like eq 4a
### plan
# 1. multivariable input scalar output (multiple times?)
# 2. 
# 3. true SINDy

# scalar multivariable case
# vector multivariable (x1,x2,y1,y2 etc)