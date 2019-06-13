
function create_splittable_from_params(params::cluster_parameters, α::Float64)
    params_l = deepcopy(params)
    params_l.distribution = sample_distribution(params.posterior_hyperparams)
    params_r = deepcopy(params)
    params_r.distribution = sample_distribution(params.posterior_hyperparams)
    lr_weights = rand(Dirichlet([α / 2, α / 2]))
    return splittable_cluster_params(params,params_l,params_r,lr_weights, false, [-Inf,-Inf,-Inf,-Inf,-Inf])
end


function merge_clusters_to_splittable(cpl::cluster_parameters,cpr::cluster_parameters, α::Float64)
    suff_stats = aggregate_suff_stats(cpl.suff_statistics,cpr.suff_statistics)
    posterior_hyperparams = calc_posterior(cpl.hyperparams, suff_stats)
    lr_weights = rand(Dirichlet([cpl.suff_statistics.N + (α / 2), cpr.suff_statistics.N + (α / 2)]))
    cp = cluster_parameters(cpl.hyperparams, cpl.distribution, suff_stats, posterior_hyperparams)
    return splittable_cluster_params(cp,cpl,cpr,lr_weights, false, [-Inf,-Inf,-Inf,-Inf,-Inf])
end


function should_merge!(should_merge::AbstractArray{Float64,1}, cpl::cluster_parameters,cpr::cluster_parameters, α::Float64, final::Bool)
    new_suff = aggregate_suff_stats(cpl.suff_statistics, cpr.suff_statistics)
    cp = cluster_parameters(cpl.hyperparams, cpl.distribution, new_suff,cpl.posterior_hyperparams)
    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl.posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr.posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp.posterior_hyperparams, cp.suff_statistics)
    log_HR = (-log(α) + lgamma(α) -2*lgamma(0.5*α) + lgamma(cp.suff_statistics.N) -lgamma(cp.suff_statistics.N + α) +
        lgamma(cpl.suff_statistics.N + 0.5*α)-lgamma(cpl.suff_statistics.N)  - lgamma(cpr.suff_statistics.N) +
        lgamma(cpr.suff_statistics.N + 0.5*α)+ log_likihood- log_likihood_l- log_likihood_r)
    if (log_HR > log(rand())) || (final && log_HR > log(0.1))
        should_merge .= 1
    end
end


function sample_cluster_params(params::splittable_cluster_params, α::Float64, first::Bool)
    points_count = Vector{Float64}()
    params.cluster_params.distribution = sample_distribution(first ? params.cluster_params.hyperparams : params.cluster_params.posterior_hyperparams)
    params.cluster_params_l.distribution = sample_distribution(first ? params.cluster_params_l.hyperparams : params.cluster_params_l.posterior_hyperparams)
    params.cluster_params_r.distribution = sample_distribution(first ? params.cluster_params_r.hyperparams : params.cluster_params_r.posterior_hyperparams)
    push!(points_count, params.cluster_params_l.suff_statistics.N)
    push!(points_count, params.cluster_params_r.suff_statistics.N)
    points_count .+= α / 2
    params.lr_weights = rand(Dirichlet(points_count))

    log_likihood_l = log_marginal_likelihood(params.cluster_params_l.hyperparams,params.cluster_params_l.posterior_hyperparams, params.cluster_params_l.suff_statistics)
    log_likihood_r = log_marginal_likelihood(params.cluster_params_r.hyperparams,params.cluster_params_r.posterior_hyperparams, params.cluster_params_r.suff_statistics)

    params.logsublikelihood_hist[1:4] = params.logsublikelihood_hist[2:5]
    params.logsublikelihood_hist[5] = log_likihood_l + log_likihood_r
    logsublikelihood_now = 0.0
    for i=1:5
        logsublikelihood_now += params.logsublikelihood_hist[i] *0.20
    end
    if logsublikelihood_now != -Inf && logsublikelihood_now - params.logsublikelihood_hist[5] < 1e-2 # propogate abs change to other versions?
        params.splittable = true
    end

    return params
end
