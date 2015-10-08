# Random Maclaurin Feature Maps
# from: Kar, Karnick, "Random Feature Maps for Dot Product Kernels", AISTATS 2012
#
# Danny Perry (dperry@cs.utah.edu)
# Nov 2014
#
# Polynomial Kernel Approximation:  (alpha * x'*y + c)^d


# @param alpha - polynomail kernel parameter multiplier
# @param c - polynomail kernel parameter offset
# @param d - polynomail kernel parameter power
# @param dimension - dimension of input data
# @param num_functions - how many functions to generate
function GenerateFunctionsPolynomial(sigma, input_dimension, num_functions)
	# draw directions from a normal distribution with appropriate sigma corresponding to the kernel:
	w = randn(num_functions, input_dimension) * (1/sigma)
	# draw random offsets from a uniform distribution in [0,2*pi]
	b = rand(num_functions) * 2 * pi

	# construct the random function struct:
	RandomMaclaurinFunctions(w,b)
end
