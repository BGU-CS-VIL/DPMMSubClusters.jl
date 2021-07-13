"""
    niw_hyperparams(κ::Float32, m::AbstractArray{Float32}, ν::Float32, ψ::AbstractArray{Float32})

[Normal Inverse Wishart](https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution)
"""
struct niw_hyperparams <: distribution_hyper_params
    κ::Float32
    m::AbstractArray{Float64}
    ν::Float32
    ψ::AbstractArray{Float64}
end

mutable struct niw_sufficient_statistics <: sufficient_statistics
    N::Float32
    points_sum::AbstractArray{Float64,1}
    S::AbstractArray{Float64,2}
end


function calc_posterior(prior:: niw_hyperparams, suff_statistics::Vector{Tuple{sufficient_statistics,Number}})
    if length(suff_statistics) == 0
        return prior
    end
    points_num = sum([post_kernel(x[2],global_time)*x[1].N for x in suff_statistics])
    if points_num == 0
        return prior
    end
    κ = prior.κ + points_num
    ν = prior.ν + points_num
    m = (prior.m.*prior.κ + sum([post_kernel(x[2],global_time)*x[1].points_sum for x in suff_statistics],dims=1)[1]) / κ
    ψ = (prior.ν * prior.ψ + prior.κ*prior.m*prior.m' -κ*m*m'+ sum([post_kernel(x[2],global_time)*x[1].S for x in suff_statistics])) / ν
    ψ = Matrix(Symmetric(ψ))
    ψ = (ψ+ψ')/2
    return niw_hyperparams(κ,m,ν,ψ)
end


function sample_distribution(hyperparams::niw_hyperparams)
    Σ = rand(Distributions.InverseWishart(hyperparams.ν, hyperparams.ν* hyperparams.ψ))
    μ = rand(Distributions.MvNormal(hyperparams.m, Σ/hyperparams.κ))
    invΣ = inv(Σ)
    chol = cholesky(Hermitian(invΣ))
    return mv_gaussian(μ,Σ,invΣ,logdet(Σ),chol.U)
end

function create_sufficient_statistics(hyper::niw_hyperparams,posterior::niw_hyperparams,points::AbstractArray{Float32,2}, pts_to_group = 0)
    if size(points,2) == 0
        return niw_sufficient_statistics(size(points,2),zeros(Float32,length(hyper.m)),zeros(Float32,length(hyper.m),length(hyper.m)))
    end
    pts = Array{Float64}(points)
    points_sum = sum(pts, dims = 2)[:]
    S = pts * pts'
    S = 0.5*(S+S')
    return niw_sufficient_statistics(size(points,2),points_sum,S)
end

function log_marginal_likelihood(hyper::niw_hyperparams, posterior_hyper::niw_hyperparams, suff_stats::Vector{Tuple{sufficient_statistics,Number}})
    D = length(suff_stats[end][1].points_sum)
    logpi = log(pi)
    return -sum([post_kernel(x[2],global_time)*x[1].N for x in suff_stats])*D*0.5*logpi +
        log_multivariate_gamma(posterior_hyper.ν/2, D)-
        log_multivariate_gamma(hyper.ν/2, D) +
         (hyper.ν/2)*(D*log(hyper.ν)+logdet(hyper.ψ))-
         (posterior_hyper.ν/2)*(D*log(posterior_hyper.ν) + logdet(posterior_hyper.ψ)) +
         (D/2)*(log(hyper.κ/posterior_hyper.κ))
end

function aggregate_suff_stats(suff_l::niw_sufficient_statistics, suff_r::niw_sufficient_statistics)
    return niw_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum, suff_l.S+suff_r.S)
end
