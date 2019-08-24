"""
    multinomial_hyper(α::AbstractArray{Float32,1})

[Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
"""
struct multinomial_hyper <: distribution_hyper_params
    α::AbstractArray{Float32,1}
end

mutable struct multinomial_sufficient_statistics <: sufficient_statistics
    N::Float32
    points_sum::AbstractArray{Float32,1}
end


function calc_posterior(prior:: multinomial_hyper, suff_statistics::multinomial_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    return multinomial_hyper(prior.α + suff_statistics.points_sum)
end

function sample_distribution(hyperparams::multinomial_hyper)
    return multinomial_dist(log.(rand(Dirichlet(Float64.(hyperparams.α)))))
end

function create_sufficient_statistics(hyper::multinomial_hyper,posterior::multinomial_hyper,points::AbstractArray{Float32,2}, pts_to_group = 0)
    pts = copy(points)
    points_sum = sum(pts, dims = 2)[:]
    S = pts * pts'
    return multinomial_sufficient_statistics(size(points,2),points_sum)
end

function log_marginal_likelihood(hyper::multinomial_hyper, posterior_hyper::multinomial_hyper, suff_stats::multinomial_sufficient_statistics)
    D = length(suff_stats.points_sum)
    logpi = log(pi)
    val = lgamma(sum(hyper.α)) -lgamma(sum(posterior_hyper.α)) + sum(lgamma.(posterior_hyper.α) - lgamma.(hyper.α))
    return val
end

function aggregate_suff_stats(suff_l::multinomial_sufficient_statistics, suff_r::multinomial_sufficient_statistics)
    return multinomial_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum)
end
