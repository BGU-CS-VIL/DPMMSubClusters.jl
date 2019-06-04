using NPZ
using Distributions
using LinearAlgebra

function generate_sph_gaussian_data(N::Int64, D::Int64, K::Int64)
	x = randn(D,N)
	tpi = rand(Dirichlet(ones(K)))
	tzn = rand(Multinomial(N,tpi))
	tz = zeros(N)

	tmean = zeros(D,K)
	tcov = zeros(D,D,K)

	ind = 1
	println(tzn)
	for i=1:length(tzn)
		indices = ind:ind+tzn[i]-1
		tz[indices] .= i
		tmean[:,i] .= rand(MvNormal(zeros(D), 100*Matrix{Float64}(I, D, D)))
		tcov[:,:,i] .= rand(InverseGamma((D+2)/2,1))*Matrix{Float64}(I, D, D)
		d = MvNormal(tmean[:,i], tcov[:,:,i])
		for j=indices
			x[:,j] = rand(d)
		end
		ind += tzn[i]
	end
	x, tz, tmean, tcov
end


function generate_gaussian_data(N::Int64, D::Int64, K::Int64)
	x = randn(D,N)
	tpi = rand(Dirichlet(ones(K)))
	tzn = rand(Multinomial(N,tpi))
	tz = zeros(N)

	tmean = zeros(D,K)
	tcov = zeros(D,D,K)

	ind = 1
	println(tzn)
	for i=1:length(tzn)
		indices = ind:ind+tzn[i]-1
		tz[indices] .= i
		tmean[:,i] .= rand(MvNormal(zeros(D), 100*Matrix{Float64}(I, D, D)))
		tcov[:,:,i] .= rand(InverseWishart(D+2, Matrix{Float64}(I, D, D)))
		d = MvNormal(tmean[:,i], tcov[:,:,i])
		for j=indices
			x[:,j] = rand(d)
		end
		ind += tzn[i]
	end
	x, tz, tmean, tcov
end


function generate_mnmm_data(N::Int64, D::Int64, K::Int64, trials::Int64)
	clusters = zeros(D,K)
	x = zeros(D,N)
	labels = rand(1:K,(N,))
	for i=1:K
		alphas = rand(1:20,(D,))
		alphas[i] = rand(30:100)
		clusters[:,i] = rand(Dirichlet(alphas))
	end
	for i=1:N
		x[:,i] = rand(Multinomial(trials,clusters[:,labels[i]]))
	end
	return x, labels, clusters
end
