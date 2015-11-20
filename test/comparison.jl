# Random Fourier Features
# from: Rahimi,Recht, "Random Features for Large-Scale Kernel Machines", NIPS 2007
#
# Danny Perry (dperry@cs.utah.edu)
# Oct 2014
#
# compare to Gaussian kernel:

using PyPlot
using RDatasets
using MAT

using RandomFourierFeatures
using Kernel
using Debug


X = []
Labels = []

#name = "iris"
#name = "baseball"
#name = "cpu"
name = "synthetic"
#name = "mnist"

if name == "iris"
	iris = dataset("datasets","iris")

	X = [ convert(Array{Float64,1}, iris[1]) convert(Array{Float64,1}, iris[2]) convert(Array{Float64,1}, iris[3]) convert(Array{Float64,1}, iris[4]) ]
	n = size(X,1)
	println("X: ", size(X))
	Labels = ASCIIString[]
	for i=1:n
		push!(Labels, iris[5][i])
	end                                                                                                                                            
elseif name == "mnist"

	f = matopen("mnist/nitish/mnist/train.mat")
	X = read(f, "train")
	close(f)
	f = matopen("mnist/nitish/mnist/test.mat")
	Xtest = read(f, "test")
	close(f)

	if true
		n = 1000
		nt = 100

		Xtmp = X
		X =Xtmp[1:n,:]
		for i=n+1:size(Xtmp,1)
			r = int(ceil(rand()*i))
			if r <= n
				X[r,:] = Xtmp[i,:]
			end
		end
		Xtmp = Xtest
		Xtest =Xtmp[1:nt,:]
		for i=nt+1:size(Xtmp,1)
			r = int(ceil(rand()*i))
			if r <= nt
				Xtest[r,:] = Xtmp[i,:]
			end
		end

		X = [X; Xtest]
	end


	#Xtest = X[1:n/2,:]
	#X = X[n/2+1:end,:]

	println("X: ", size(X))
	#println("Xtest: ", size(Xtest))


elseif name == "cpu"

	f = matopen("nystrom_rff_paper/cpu/cpu.mat")
	X = read(f, "Xtrain")
	X = X'

	if true
		#n = 1021
		#n = 3000 * 2
		n = 1200 
		#n = 121*2
		Xtmp = X
		X = zeros(n, size(Xtmp,2))
		for i=1:n
			X[i,:] = Xtmp[i,:]
		end
		for i=n+1:size(Xtmp,1)
			r = int(ceil(rand()*i))
			if r <= n
				X[r,:] = Xtmp[i,:]
			end
		end
	end


	#Xtest = X[1:n/2,:]
	#X = X[n/2+1:end,:]

	println("X: ", size(X))
	#println("Xtest: ", size(Xtest))

elseif name == "synthetic"

	#f = matopen("SDU_20000_1000_500_10.mat")
	f = matopen("SDU_1000_100_50_10.mat")
	X = read(f, "data")
	#Xtest = X[end-1000:end,:]
	#X = X[1:end-1000-1,:]

	n = 1000

	if n > 0
		Xtmp = X
		X = Xtmp[1:n,:]
		for i=n+1:size(Xtmp,1)
			r = int(ceil(rand()*i))
			if r <= n
				X[r,:] = Xtmp[i,:]
			end
		end
	end

	nt = 0
	if nt > 0
		Xtmp = Xtest
		Xtest = Xtmp[1:nt,:]
		for i=n+1:size(Xtmp,1)
			r = int(ceil(rand()*i))
			if r <= n
				Xtest[r,:] = Xtmp[i,:]
			end
		end
	end

	println("X: ", size(X))

else

	baseball = dataset("plyr","baseball")

	numerical_cols = [1,4,8,9,10,11,12,13,17]
	X = zeros(size(baseball,1),length(numerical_cols))

	for i=1:length(numerical_cols)
		println(i,": ",numerical_cols[i])
		X[:,i] = convert(Array{Float64,1}, baseball[numerical_cols[i]])
	end

end

n = size(X,1)
X_centered = X .- (ones(n,1)*mean(X,1))

# compute neighborhood distances to inform choice of sigma:
#sig2 = 1e9
#sig2 = 10
sig2 = 10 # TODO: try some different data sets with varying sigma...
if true

	sigma = 0
	for i=1:n
		sigma += sum( sqrt( sum( (X .- ones(n,1)*X[i,:]).^2, 2 ) ) )
	end
	sigma /= n^2

	sig2 = sigma^2

	figure()
	for i=1:n
		x = X[i,:]
		d2 = sum((X - ones(n,1)*x).^2,2)
		#println(i,": ", sort(vec(d2))[end-4:end])
		plot(1:length(d2),sort(vec(d2)),"--")
		hold(:on)
	end
	plot(1:n,sig2*ones(n),".-")
	title("distance squared to other points")
	ylabel("distance squared")
	xlabel("sorted neighbors")
end


# generate the standard guassian kernel:
#sigma = sqrt(sig2) * 1e-2
paramstr = @sprintf("s=%.1f",sigma)


# now generate a kernel matrix from the approximate space:
function comparison(X,Xtest,sigma)

	println("computing kernel matrix for param: ",paramstr," ...")
	K = GaussianKernel(Xtest,Xtest,sigma)
	Ktrain = GaussianKernel(X,Xtest,sigma)

	println("starting comparison")
	
	inputdim = size(X,2)
	dims = Int64[]
	errs = Float64[]
	errs2 = Float64[]
	maxerrs = Float64[]
	minerrs = Float64[]
	errs_ff = Float64[]
	errs_nystrom = Float64[]
	maxerrs_nystrom = Float64[]
	minerrs_nystrom = Float64[]
	#figure()
	#for dim=10:100:1000
	for l=10:10:100
		dim = inputdim*l
		#dim = l
		println("dim: ",dim)
		err = 0
		err2 = 0
		maxerr = 0
		minerr = 0
		err_ff = 0
		err_nystrom = 0
		maxerr_nystrom = 0
		minerr_nystrom = 0
		samples = 10
		for i=1:samples
			# random fourier features
			rffs = GenerateFunctionsGaussian(typeof(Xtest[1,1]), sigma, inputdim, dim)
			rffs2 = GenerateFunctionsGaussianSinCos(typeof(Xtest[1,1]), sigma, inputdim, dim)
			Xp = RFFProject( Xtest, rffs )
			Xp2 = RFFProjectSinCos( Xtest, rffs2 )
			Kp = Xp * Xp'
			Kp2 = Xp2 * Xp2'
			#absdiff = abs(K[:] .- Kp[:])
			#absdiff2 = abs(K[:] .- Kp2[:])
			absdiff = vecnorm(K - Kp) #/ length(K)
			absdiff2 = vecnorm(K - Kp2) #/ length(K)
			err += absdiff
			err2 += absdiff2
			maxerr += maximum(absdiff)
			minerr += minimum(absdiff)
			# fast food features
			ffs = GenerateFastFood(Float64, dim, inputdim)
			Xff = FFProject( Xtest', ffs, sigma )
			Kff = Xff' * Xff
			err_ff += vecnorm(K-Kff)

			# random sub matrix (Nystrom method)
			# reservoir subsample:
			nystrom_num = min(size(X,1), dim)
			if false
				subset = [1:nystrom_num]
				for j=nystrom_num+1:size(X,1)
					r = ceil(rand()*j)
					if r <= nystrom_num
						subset[r] = j
					end
				end
			else
				subset = int(ceil(rand(nystrom_num) * size(X,1)))
			end
			K_subset = GaussianKernel(X[subset,:], sigma)
			#K_all = GaussianKernel(X, X_subset, sigma)
			K_all = Ktrain[subset,:]
			if cond(K_subset) > 1e10
				UU,SS,VV = svd(K_subset)
				SS = diagm(SS) + 1e-10*eye(length(SS))
				K_subset = UU*SS*VV'
			end
			K_nystrom = K_all' * (K_subset \ K_all)
			absdiff_nystrom = vecnorm(K - K_nystrom) #/ length(K)
			err_nystrom += mean(absdiff_nystrom)
			maxerr_nystrom += maximum(absdiff_nystrom)
			minerr_nystrom += minimum(absdiff_nystrom)

		end
		err /= samples
		err2 /= samples
		maxerr /= samples
		minerr /= samples

		err_ff /= samples

		err_nystrom /= samples
		maxerr_nystrom /= samples
		minerr_nystrom /= samples
		#println("avg err: ", err )
		
		push!(dims, dim)
		push!(errs, err)
		push!(errs2, err2)
		push!(minerrs, minerr)
		push!(maxerrs, maxerr)
		
		push!(errs_ff, err_ff)

		push!(errs_nystrom, err_nystrom)
		push!(minerrs_nystrom, minerr_nystrom)
		push!(maxerrs_nystrom, maxerr_nystrom)

	end

	figure()
	plot(dims,errs,"r-")
	hold(:on)
	plot(dims,errs2,"b-")
	plot(dims,errs_ff,"g:")
	plot(dims,errs_nystrom,"k-")
	legend(["RFF avg err","RFF 2 avg err", "FF avg err", "Nys avg err"],loc="upper right")
	title("RR dim vs avg error")

	figure()
	plot(dims,errs,"r-")
	hold(:on)
	plot(dims,errs2,"b-")
	plot(dims,errs_ff,"g:")
	plot(dims,errs_nystrom,"k-")
	xscale("log")
	yscale("log")
	legend(["RFF avg err","RFF 2 avg err", "FF avg err", "Nys avg err"],loc="lower left")
	title("Log scale: RR dim vs avg error")


	PyPlot.plt[:show]()
	#@bp

end

comparison(X_centered[1:end-100,:], X_centered[end-100+1:end,:], sigma)
