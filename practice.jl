# Sparse representation algorithm
#####################################
f0(x) = 1
f1(x) = x
f2(x) = x^2
f3(x) = x^3
f4(y) = (2y)
f5(y) = (2y^2)
f6(y) = (2y^3)
f7(x) = f1(x) * f4(x)
f8(x) = f1(x) * f5(x)
f9(x) = f2(x) * f4(x)
f10(x) = f2(x) * f5(x)

library = [f0, f1, f2, f4, f7]
times = [i for i = 1:10]
vars = 2
data = [
    0 2 6 12 20 30 42 56 72 90;
    5 11 21 35 53 75 101 131 165 203
] # 1. x^2-x
# 2. xy + 3 (x = t, y = 2t)
function sparse_representation(library, data, times, vars)
    helper = []
    n_iterations = 10
    lambda = 0.25

    for i in 1:vars-1
        times = [times; times]
    end

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
    theta_arrays = []

    if vars == 2
        midpoint = Integer(length(data) / 2)
        row1 = data[1:midpoint]
        row2 = data[midpoint+1:length(data)]
        row3 = theta[1:midpoint, :]
        row4 = theta[(midpoint+1:length(data)), :]
        data_arrays = [[row1] [row2]]
        theta_arrays = [[row3] [row4]]
    elseif vars == 1
        data_arrays = data
        theta_arrays = theta
    end


    for l in 1:vars
        Xi = theta_arrays[l] \ data_arrays[l]
        # print(Xi)
        for k in 1:n_iterations
            smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
            # println(smallinds)
            Xi[smallinds] .= 0
            biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
            # println(biginds)
            Xi[biginds] = theta_arrays[l][:, biginds] \ data_arrays[l]
        end
        println(l, ": ", Xi)
    end

end

theta = sparse_representation(library, data, times, vars)


########### Scalar Multivariable ####################
# data = [
#     0 3 16 45 96 175 288 441 640 891 1200 1573 2016 2535 3136 3825 4608 5491 6480 7581 8800
# ] # xy + x^3 (x = t, y =  2t)
# times = [i for i = 0:20]
# row_1 = [1 for t in times]
# row_x = [f1(t) for t in times]
# row_y = [f4(2t) for t in times]
# row_xy = [f1(t) * f4(2t) for t in times]
# row_x3 = [f3(t) for t in times]
# theta = [row_1;; row_x;; row_y;; row_xy;; row_x3]
# lambda = 0.25
# data = vec(permutedims(data)) # Vector of data, x then y
# Xi = theta \ data
# for k in 1:n_iterations
#     smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
#     # println(smallinds)
#     Xi[smallinds] .= 0
#     biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
#     # println(biginds)
#     Xi[biginds] = theta[:, biginds] \ data
# end
# println(Xi)
#####################################################
########### Vector Multivariable ####################
data = [
    0 3 16 45 96 175 288 441 640 891 1200;
    3 5 11 21 35 53 75 101 131 165 203
] # 1. xy + x^3, 
# 2. xy + 3 (x = t, y = 2t)

times = [[i for i = 0:10]; [i for i = 0:10]]
row_1 = [1 for t in times]
row_x = [f1(t) for t in times]
row_y = [f4(t) for t in times]
row_xy = [f1(t) * f4(t) for t in times]
row_x3 = [f3(t) for t in times]

theta = [row_1;; row_x;; row_y;; row_xy;; row_x3]

lambda = 0.25
data = vec(permutedims(data))
data1 = data[1:11]
data2 = data[12:22]
theta1 = theta[1:11, :]
theta2 = theta[12:22, :]
datas = [[data1] [data2]]
thetas = [[theta1] [theta2]]
for l in 1:size(datas, 2)
    Xi = thetas[l] \ datas[l]
    for k in 1:n_iterations
        smallinds = findall(p -> (p < abs(lambda)), Xi) #array of indicies with small coefficients
        # println(smallinds)
        Xi[smallinds] .= 0
        biginds = [i for i = 1:length(Xi) if !(i in smallinds)]
        # println(biginds)
        Xi[biginds] = thetas[l][:, biginds] \ datas[l]
    end
    println(l, ": ", Xi)
end
######################################################
###################################### make theta vector of matricies w/ each matrix like eq 4a
### plan
# 1. clean up, make funtion with few inputs (data & library)
# 2. 
# 3. true SINDy

# scalar multivariable case
# vector multivariable (x1,x2,y1,y2 etc)