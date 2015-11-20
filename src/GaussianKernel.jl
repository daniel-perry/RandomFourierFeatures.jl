# Random Fourier Features
# from: Rahimi,Recht, "Random Features for Large-Scale Kernel Machines", NIPS 2007
#
# Danny Perry (dperry@cs.utah.edu)
# Oct 2014
#
# Gaussian Kernel Approximation

using Debug
using Hadamard
using Distributions

# A set of random fourier functions
type RandomFourierFunctions{T}
	directions::Array{T,2} # random direction
	offsets::Array{T,1} # random shift in [0,2*pi]
end



# project the input data into the approximate kernel space
#
# @param X - input data
# @param RFFs - the random fourier functions
function RFFProject!{T}(z::Array{T,2}, X::Array{T,2}, RFFs::RandomFourierFunctions{T})
	n = size(X,1)
	D = size(RFFs.directions,1)
	z[:,:] = convert(Array{T,2}, (sqrt(2/D)) * cos( (X * RFFs.directions') .+ ones(n,1) * RFFs.offsets' ) )
end
function RFFProject{T}(X::Array{T,2}, RFFs::RandomFourierFunctions{T})
	z = zeros(T, size(X,1), size(RFFs.directions,1))
	RFFProject!(z, X, RFFs)
	return z
end
# use the sin-cos form instead
function RFFProjectSinCos!{T}(z::Array{T,2}, X::Array{T,2}, RFFs::RandomFourierFunctions{T})
	n = size(X,1)
	D = 2*size(RFFs.directions,1)
	z[:,2:2:end] = X * RFFs.directions'
	z[:,1:2:end] = convert( Array{T,2}, sin( z[:,2:2:end] ) )
	z[:,2:2:end] = convert( Array{T,2}, cos( z[:,2:2:end] ) )
	z[:,:] *= (sqrt(2/D))
end
function RFFProjectSinCos{T}(X::Array{T,2}, RFFs::RandomFourierFunctions{T})
	z = zeros(T, size(X,1), 2*size(RFFs.directions,1))
	RFFProjectSinCos!(z, X, RFFs)
	return z
end

# @param sigma - Gaussian kernel parameter
# @param dimension - dimension of input data
# @param num_functions - how many functions to generate
function GenerateFunctionsGaussian(T, sigma, input_dimension, num_functions)
	# draw directions from a normal distribution with appropriate sigma corresponding to the kernel:
	w = randn(num_functions, input_dimension) * (1/sigma)

	# draw random offsets from a uniform distribution in [0,2*pi]
	b = rand(T, (num_functions,)) * 2 * pi

	# construct the random function struct:
	RandomFourierFunctions{T}(w,b)
end
# use the sin-cos form instead
function GenerateFunctionsGaussianSinCos(T, sigma, input_dimension, num_functions)
	if mod(num_functions,2) != 0
		error("num_functions must be an even number.")
	end
	# draw directions from a normal distribution with appropriate sigma corresponding to the kernel:
	w = randn(div(num_functions,2), input_dimension) * (1/sigma)

	# draw random offsets from a uniform distribution in [0,2*pi]
	b = zeros(1)

	# construct the random function struct:
	RandomFourierFunctions{T}(w,b)
end



# A version that draws from the probability distribution with bias towards the data
# @param sigma - Gaussian kernel parameter
# @param dimension - dimension of input data
# @param num_functions - how many functions to generate
# @param Xsample - subsample of data to select random basis
# @param Ksample - subsample of data to select random basis as kernel matrix
# @param centered - whether or not the data should be centered
function GenerateFunctionsGaussianBiased(sigma, input_dimension, num_functions, Xsample, Ksample, centered)
	w = zeros(num_functions, input_dimension)
	b = zeros(num_functions)

	samples = num_functions
	runs = 100
	# take $(samples) and pick the closest to the eigenvectors

	for k=1:samples:num_functions
		nsamples = samples
		if k + samples > num_functions
			nsamples = (num_functions-k+1)
		end
		dists = Float64[]
		rffs = RandomFourierFunctions[]
		for i=1:runs
			ws = randn(nsamples, input_dimension) * (1/sigma)
			bs = rand(nsamples) * 2 * pi
			push!(rffs, RandomFourierFunctions(ws,bs))
			zs = RFFProject( Xsample, rffs[end] )
			if centered
				zs  = zs - ones(size(zs,1),1)*mean(zs,1)
			end
			Ktmp = zs * zs'
			#push!(dists, vecnorm(Ktmp - Ksample))
			push!(dists, norm(Ktmp - Ksample))
		end
		minval,mini = findmin(dists)
		w[k:k+nsamples-1,:] = rffs[mini].directions
		b[k:k+nsamples-1] = rffs[mini].offsets
	end

	# draw directions from a normal distribution with appropriate sigma corresponding to the kernel:
	#w = randn(num_functions, input_dimension) * (1/sigma)
	# draw random offsets from a uniform distribution in [0,2*pi]
	#b = rand(num_functions) * 2 * pi

	# construct the random function struct:
	RandomFourierFunctions(w,b)
end

# Fast Food - approximate kernel space
# adapted from matlab implementation found here:
# http://www.mathworks.com/matlabcentral/fileexchange/49142-fastfood-kernel-expansions

# A set of fast food functions
type FastFood{T}
	b::Array{Int64,2}
	g::Array{T,2}
	p::Array{Int64,2}
	s::Array{T,2}
end

# @param dimension - dimension of input data
# @param num_functions - how many functions to generate
function GenerateFastFood(T, num_functions, dimension)

	d = dimension
	n = num_functions

	l = int(ceil(log2(d)))
	d = 2^l
	k = int(ceil(n/d))
	n = d*k

	# B matrix - diagonal of uniform {-1,1} values
	b = ((rand(d,k) .> .5)*2) .- 1

	# G matrix - diagonal of Guassian values
	g = randn(d,k)

	# Π matrix - diagonal permutation matrix
	Π = zeros(Int64,d,k) #randperm(d,k)

	if true
		# S matrix - scaling matrix
		gfro = 1.0/sqrt(vecnorm(g))
		s = zeros(k,d)
		gammad = Gamma(d-1,1)
		for i=1:k
			Π[:,i] = randperm(d)
			s[i,:] = rand(gammad, d)
		end
		tmp = gfro * s

		s = zeros(n, 1);
		for i = 1:k
				s[((i-1)*d+1):(i*d)] = tmp[i]
		end
		
		return FastFood{T}(b,g,Π,s) # to go
	else
		for i=1:k
			Π[:,i] = randperm(d)
		end
	end
	
	FastFood{T}(b,g,Π) # to go
end

# @param X - input data
# @param ff - fastfood parameters
# @param sigma - Gaussian kernel parameter
function FFProject{T}(X::Array{T,2}, ff::FastFood{T}, sigma)
	if !isa(sigma,T)
		sigma = convert(T,sigma)
	end
	m = size(X,2) # num data points
	d = size(X,1) # dimension of data
	k = size(ff.b,2) # factor of num of FF dimensions

	# pad with zeros until d = 2^l
	l = int(ceil(log2(d)))
	d = 2^l
	if d == size(X,1)
		X = X
	else
		X2 = zeros(T, d,m)
		X2[1:size(X,1),:] = X
		X = X2
	end
	n = d*k

	THT = zeros(T, n,m)

	for i=1:k
		tmp = copy(X)
		for j=1:m
			tmp[:,j] .*= ff.b[:,i]
		end
		tmp = fwht_natural(tmp,1)
		tmp = tmp[ff.p[:,i],:]
		for j=1:m
			tmp[:,j] .*= ff.g[:,i]*d
		end
    ind = ((i-1)*d+1):(i*d)
		THT[ind, :] = fwht_natural(tmp,1)
	end
	for i=1:m
		THT[:,i] .*= ff.s 
	end

	THT ./= sigma
	phi = [cos(THT); sin(THT)] ./ sqrt(n)
end

