# Random Fourier Features
# from: Rahimi,Recht, "Random Features for Large-Scale Kernel Machines", NIPS 2007
#
# Danny Perry (dperry@cs.utah.edu)
# Oct 2014

module RandomFourierFeatures

export RandomFourierFunctions, GenerateFunctionsGaussian, GenerateFunctionsGaussianSinCos, GenerateFunctionsGaussianBiased, RFFProject, RFFProjectSinCos, FastFood, GenerateFastFood, FFProject

include("GaussianKernel.jl")

# A set of random fourier functions
type RandomMaclaurinFunctions
	directions::Array{Float64,2} # random direction
	offsets::Array{Float64,1} # random shift in [0,2*pi]
end

# project the input data into the approximate kernel space
#
# @param X - input data
# @param RFFs - the random fourier functions
function RMFProject(X::Array{Float64,2}, RFFs::RandomMaclaurinFunctions)
	n = size(X,1)
	z = sqrt(2) * cos( (X * RFFs.directions') .+ ones(n,1) * RFFs.offsets' ) * (1/sqrt(length(RFFs.offsets)))
end


include("PolynomialKernel.jl")

end # module
