# Sparse representation algorithm
#####################################

f1(x) = x
f2(x) = x^2
f3(x) = x^3
f4(y) = y

n_iterations = 10
n_vars = 1
# data = [
#     0 1 2 3 4 5 6 7 8 9 10;
#     0 2 4 6 8 10 12 14 16 18 20
# ] # x,2y

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
] # 1. xy + x^3, 2. xy + 3 (x = t, y = 2t)

times = [[i for i = 0:10]; [i for i = 0:10]]
row_1 = [1 for t in times]
row_x = [f1(t) for t in times]
row_y = [f4(2t) for t in times]
row_xy = [f1(t) * f4(2t) for t in times]
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
# 1. write function for 1 variable case
# 2. put ind back in to get multivariable case (vectorize)
# 3. true SINDy

# scalar multivariable case
# vector multivariable (x1,x2,y1,y2 etc)