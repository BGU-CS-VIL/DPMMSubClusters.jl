# Priors

The package currently support two priors, `NIW` for observations generated from gaussians, and `Dirichlet Distribution` for multinomial data.

Each prior is defined by the prior itself, and the distribution which is sampled from it.

## Existing Priors
### NIW (Gaussian)
For data generated from Gaussians we use a `NIW` prior, which a `Multivariate Normal` distribution is sampled from:
```@docs
DPMMSubClusters.niw_hyperparams
DPMMSubClusters.mv_gaussian
```

### Multinomial
For multinomial distribution we will a `Dirichlet Distribution` prior, and a `Categorial Distribution` is sampled from it.
```@docs
DPMMSubClusters.multinomial_hyper
DPMMSubClusters.multinomial_dist
```

## Creating new Priors

If you require to create a new prior, you need to create both a distribution file, and a prior file (each in its designated folder).
Note that the package only support conjugate priors.

The prior file must implement the following:

```julia
struct prior_hyper_params <: distribution_hyper_params
    ...
end

mutable struct prior_sufficient_statistics <: sufficient_statistics
    N::Float32 #Must have this, even if not needed!
    ...
end


function calc_posterior(prior:: prior_hyper_params, suff_statistics::prior_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    return prior_hyper_params(...)
end

function sample_distribution(hyperparams::prior_hyper_params)
    return new_dist(...)
end

function create_sufficient_statistics(hyper::prior_hyper_params,posterior::prior_hyper_params,points::AbstractArray{Float32,2}, pts_to_group = 0)
    pts = copy(points)
    ...
    return prior_sufficient_statistics(size(points,2),...)
end

function log_marginal_likelihood(hyper::prior_hyper_params, posterior_hyper::prior_hyper_params, suff_stats::prior_sufficient_statistics)
    ...
    return value
end

function aggregate_suff_stats(suff_l::prior_sufficient_statistics, suff_r::prior_sufficient_statistics)
    return prior_sufficient_statistics(...)
end
```

For the distribution file you must implement the following:
```julia
struct new_dist <: distribution_sample
    ...
end


function log_likelihood!(r::AbstractArray,x::AbstractMatrix, distribution_sample::mv_gaussian , group::Int64 = -1)
     ...
    r .= (each sample log likelihood)
end
```
