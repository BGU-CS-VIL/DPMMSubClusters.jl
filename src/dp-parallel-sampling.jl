"""
    init_model()

Initialize the model, loading the data from external `npy` files, specified in the params file.
All prior data as been included previously, and is globaly accessed by the function.

Returns an `dp_parallel_sampling` (e.g. the main data structure) with the configured parameters and data.
"""
function init_model()
    if random_seed != nothing
        @eval @everywhere seed!($random_seed)
    end
    if use_verbose
        println("Loading and distributing data:")
        @time data = distribute(Float32.(load_data(data_path, prefix = data_prefix)))
    else
        data = distribute(Float32.(load_data(data_path, prefix = data_prefix)))
    end
    total_dim = size(data,2)
    model_hyperparams = model_hyper_params(hyper_params,α,total_dim)

    labels = distribute(rand(1:initial_clusters,(size(data,2))) .+ ((outlier_mod > 0) ? 1 : 0))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))
    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float32[])
    return dp_parallel_sampling(model_hyperparams,group)
end

"""
    init_model(all_data)

Initialize the model, from `all_data`, should be `Dimensions X Samples`, type `Float32`
All prior data as been included previously, and is globaly accessed by the function.

Returns an `dp_parallel_sampling` (e.g. the main data structure) with the configured parameters and data.
"""
function init_model_from_data(all_data)
    if random_seed != nothing
        @eval @everywhere Random.seed!($random_seed)
    end
    if use_verbose
        println("Loading and distributing data:")
        @time data = distribute(all_data)
    else
        data = distribute(all_data)
    end

    total_dim = size(data,2)
    model_hyperparams = model_hyper_params(hyper_params,α,total_dim)
    labels = distribute(rand(1:initial_clusters,(size(data,2))) .+ ((outlier_mod > 0) ? 1 : 0))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))
    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float32[])
    return dp_parallel_sampling(model_hyperparams,group)
end

"""
    init_first_clusters!(dp_model::dp_parallel_sampling, initial_cluster_count::Int64))

Initialize the first clusters in the model, according to the number defined by initial_cluster_count

Mutates the model.
"""
function init_first_clusters!(dp_model::dp_parallel_sampling, initial_cluster_count::Int64)
    if outlier_mod > 0
        push!(dp_model.group.local_clusters, create_outlier_local_cluster(dp_model.group,outlier_hyper_params))
    end
    for i=1:initial_cluster_count
        push!(dp_model.group.local_clusters, create_first_local_cluster(dp_model.group))
    end
    @sync update_suff_stats_posterior!(dp_model.group)
    sample_clusters!(dp_model.group,false)
    broadcast_cluster_params([create_thin_cluster_params(x) for x in dp_model.group.local_clusters],[1.0])
end


"""
    dp_parallel(all_data::AbstractArray{Float32,2},
        local_hyper_params::distribution_hyper_params,
        α_param::Float32,
         iters::Int64 = 100,
         init_clusters::Int64 = 1,
         seed = nothing,
         verbose = true,
         save_model = false,
         burnout = 15,
         gt = nothing,
         max_clusters = Inf,
         outlier_weight = 0,
         outlier_params = nothing)

Run the model.
# Args and Kwargs
 - `all_data::AbstractArray{Float32,2}` a `DxN` array containing the data
 - `local_hyper_params::distribution_hyper_params` the prior hyperparams
 - `α_param::Float32` the concetration parameter
 - `iters::Int64` number of iterations to run the model
 - `init_clusters::Int64` number of initial clusters
 - `seed` define a random seed to be used in all workers, if used must be preceeded with `@everywhere using random`.
 - `verbose` will perform prints on every iteration.
 - `save_model` will save a checkpoint every 25 iterations.
 - `burnout` how long to wait after creating a cluster, and allowing it to split/merge
 - `gt` Ground truth, when supplied, will perform NMI and VI analysis on every iteration.
 - `max_clusters` limit the number of cluster
 - `outlier_weight` constant weight of an extra non-spliting component
 - `outlier_params` hyperparams for an extra non-spliting component

# Return values
dp_model, iter_count , nmi_score_history, liklihood_history, cluster_count_history
 - `dp_model` The DPMM model inferred
 - `iter_count` Timing for each iteration
 - `nmi_score_history` NMI score per iteration (if gt suppled)
 - `likelihood_history` Log likelihood per iteration.
 - `cluster_count_history` Cluster counts per iteration.
"""
function dp_parallel(all_data::AbstractArray{Float32,2},
        local_hyper_params::distribution_hyper_params,
        α_param::Float32,
         iters::Int64 = 100,
         init_clusters::Int64 = 1,
         seed = nothing,
         verbose = true,
         save_model = false,
         burnout = 15,
         gt = nothing,
         max_clusters = Inf,
         outlier_weight = 0,
         outlier_params = nothing)
    global iterations = iters
    global random_seed = seed
    global hyper_params = local_hyper_params
    global initial_clusters = init_clusters
    global α = α_param
    global use_verbose = verbose
    global should_save_model = save_model
    global burnout_period = burnout
    global max_num_of_clusters = max_clusters
    global outlier_mod = outlier_weight
    global outlier_hyper_params = outlier_params
    dp_model = init_model_from_data(all_data)
    global leader_dict = get_node_leaders_dict()
    init_first_clusters!(dp_model, initial_clusters)
    if use_verbose
        println("Node Leaders:")
        println(leader_dict)
    end
    global ground_truth = gt
    @eval @everywhere global hard_clustering = $hard_clustering
    return run_model(dp_model, 1)
end

"""
    fit(all_data::AbstractArray{Float32,2},local_hyper_params::distribution_hyper_params,α_param::Float32;
       iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false, burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing)

Run the model (basic mode).
# Args and Kwargs
 - `all_data::AbstractArray{Float32,2}` a `DxN` array containing the data
 - `local_hyper_params::distribution_hyper_params` the prior hyperparams
 - `α_param::Float32` the concetration parameter
 - `iters::Int64` number of iterations to run the model
 - `init_clusters::Int64` number of initial clusters
 - `seed` define a random seed to be used in all workers, if used must be preceeded with `@everywhere using random`.
 - `verbose` will perform prints on every iteration.
 - `save_model` will save a checkpoint every 25 iterations.
 - `burnout` how long to wait after creating a cluster, and allowing it to split/merge
 - `gt` Ground truth, when supplied, will perform NMI and VI analysis on every iteration.
 - `max_clusters` limit the number of cluster
 - `outlier_weight` constant weight of an extra non-spliting component
 - `outlier_params` hyperparams for an extra non-spliting component

# Return Values
 - `labels` Labels assignments
 - `clusters` Cluster parameters
 - `weights` The cluster weights, does not sum to `1`, but to `1` minus the weight of all uninstanistaed clusters.
 - `iter_count` Timing for each iteration
 - `nmi_score_history` NMI score per iteration (if gt suppled)
 - `likelihood_history` Log likelihood per iteration.
 - `cluster_count_history` Cluster counts per iteration.
 - `sub_labels` Sub labels assignments

# Example:
```julia
julia> x,y,clusters = generate_gaussian_data(10000,2,6,100.0)
...

julia> hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
                  zeros(2),
                  5,
                  [1 0;0 1])
DPMMSubClusters.niw_hyperparams(1.0f0, Float32[0.0, 0.0], 5.0f0, Float32[1.0 0.0; 0.0 1.0])

julia> ret_values= fit(x,hyper_params,10.0, iters = 100, verbose=false)

...

julia> unique(ret_values[1])
6-element Array{Int64,1}:
 3
 6
 1
 2
 5
 4
```
"""
function fit(all_data::AbstractArray{Float32,2},local_hyper_params::distribution_hyper_params,α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false, burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing)
        dp_model,iter_count , nmi_score_history, liklihood_history, cluster_count_history = dp_parallel(all_data, local_hyper_params,α_param,iters,init_clusters,seed,verbose, save_model,burnout,gt, max_clusters, outlier_weight, outlier_params)
        return Array(dp_model.group.labels), [x.cluster_params.cluster_params.distribution for x in dp_model.group.local_clusters], dp_model.group.weights,iter_count , nmi_score_history, liklihood_history, cluster_count_history,Array(dp_model.group.labels_subcluster)
end

"""
    fit(all_data::AbstractArray{Float32,2},α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false,burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing)


Run the model (basic mode) with default `NIW` prior.
# Args and Kwargs
 - `all_data::AbstractArray{Float32,2}` a `DxN` array containing the data
 - `α_param::Float32` the concetration parameter
 - `iters::Int64` number of iterations to run the model
 - `init_clusters::Int64` number of initial clusters
 - `seed` define a random seed to be used in all workers, if used must be preceeded with `@everywhere using random`.
 - `verbose` will perform prints on every iteration.
 - `save_model` will save a checkpoint every 25 iterations.
 - `burnout` how long to wait after creating a cluster, and allowing it to split/merge
 - `gt` Ground truth, when supplied, will perform NMI and VI analysis on every iteration.
 - `outlier_weight` constant weight of an extra non-spliting component
 - `outlier_params` hyperparams for an extra non-spliting component

# Return Values
 - `labels` Labels assignments
 - `clusters` Cluster parameters
 - `weights` The cluster weights, does not sum to `1`, but to `1` minus the weight of all uninstanistaed clusters.
 - `iter_count` Timing for each iteration
 - `nmi_score_history` NMI score per iteration (if gt suppled)
 - `likelihood_history` Log likelihood per iteration.
 - `cluster_count_history` Cluster counts per iteration.
 - `sub_labels` Sub labels assignments

# Example:
```julia
julia> x,y,clusters = generate_gaussian_data(10000,2,6,100.0)
...

julia> ret_values= fit(x,10.0, iters = 100, verbose=false)

...

julia> unique(ret_values[1])
6-element Array{Int64,1}:
 3
 6
 1
 2
 5
 4
```
"""
function fit(all_data::AbstractArray{Float32,2},α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false,burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing)
    data_dim = size(all_data,1)
    cov_mat = Matrix{Float32}(I, data_dim, data_dim)
    local_hyper_params = niw_hyperparams(1,zeros(Float32,data_dim),data_dim+3,cov_mat)
    dp_model,iter_count , nmi_score_history, liklihood_history, cluster_count_history = dp_parallel(all_data, local_hyper_params,α_param,iters,init_clusters, seed,verbose,save_model,burnout,gt, max_clusters,outlier_weight, outlier_params)
    return Array(dp_model.group.labels), [x.cluster_params.cluster_params.distribution for x in dp_model.group.local_clusters], dp_model.group.weights,iter_count , nmi_score_history, liklihood_history, cluster_count_history, Array(dp_model.group.labels_subcluster)
end

fit(all_data::AbstractArray, α_param;
        iters = 100, init_clusters = 1,
        seed = nothing, verbose = true,
        save_model = false,burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing) =
    fit(Float32.(all_data),Float32(α_param),iters = Int64(iters),
        init_clusters=Int64(init_clusters), seed = seed, verbose = verbose,
        save_model = save_model, burnout = burnout, gt = gt, max_clusters = max_clusters, outlier_weight = outlier_weight, outlier_params = outlier_params)

fit(all_data::AbstractArray,local_hyper_params::distribution_hyper_params,α_param;
        iters = 100, init_clusters::Number = 1,
        seed = nothing, verbose = true,
        save_model = false,burnout = 20, gt = nothing, max_clusters = Inf, outlier_weight = 0, outlier_params = nothing) =
    fit(Float32.(all_data),local_hyper_params,Float32(α_param),iters = Int64(iters),
        init_clusters=Int64(init_clusters), seed = seed, verbose = verbose,
        save_model = save_model, burnout = burnout, gt = gt, max_clusters = max_clusters, outlier_weight = outlier_weight, outlier_params = outlier_params)




"""
    dp_parallel(model_params::String; verbose = true, save_model = true,burnout = 5, gt = nothing)

Run the model in advanced mode.
# Args and Kwargs
 - `model_params::String` A path to a parameters file (see below)
 - `verbose` will perform prints on every iteration.
 - `save_model` will save a checkpoint every `X` iterations, where `X` is specified in the parameter file.
 - `burnout` how long to wait after creating a cluster, and allowing it to split/merge
 - `gt` Ground truth, when supplied, will perform NMI and VI analysis on every iteration.

# Return values
dp_model, iter_count , nmi_score_history, liklihood_history, cluster_count_history
 - `dp_model` The DPMM model inferred
 - `iter_count` Timing for each iteration
 - `nmi_score_history` NMI score per iteration (if gt suppled)
 - `likelihood_history` Log likelihood per iteration.
 - `cluster_count_history` Cluster counts per iteration.
"""
function dp_parallel(model_params::String; verbose = true, gt = nothing)
    include(model_params)
    global use_verbose = verbose
    dp_model = init_model()
    global leader_dict = get_node_leaders_dict()
    global should_save_model = enable_saving
    global ground_truth = gt
    global burnout_period = burnout_period
    global max_num_of_clusters = max_clusters
    init_first_clusters!(dp_model, initial_clusters)
    if use_verbose
        println("Node Leaders:")
        println(leader_dict)
    end
    @eval @everywhere global hard_clustering = $hard_clustering
    return run_model(dp_model, 1 ,model_params)
end

function run_model(dp_model, first_iter, model_params="none", prev_time = 0)
    start_time= time()
    iter_count = []
    liklihood_history = []
    v_score_history = []
    nmi_score_history = []
    global ground_truth
    cur_parr_count = 10
    cluster_count_history = []

    @sync for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i set_parr_worker(dp_model.group.labels,cur_parr_count)
    end


    for i=first_iter:iterations
        # plot_group(dp_model.group)

        final = false
        no_more_splits = false
        if i >= iterations - argmax_sample_stop #We assume the cluters k has been setteled by now, and a low probability random split can do dmg
            final = true
        end
        if i >= iterations - split_stop || length(dp_model.group.local_clusters) >= max_num_of_clusters
            no_more_splits = true
        end

        prev_time = time()
        group_step(dp_model.group, no_more_splits, final, i==1)
        iter_time = time() - prev_time
        push!(iter_count,iter_time)

        push!(cluster_count_history,length(dp_model.group.local_clusters))

        if ground_truth != nothing
            group_labels = Array(dp_model.group.labels)
            push!(v_score_history, varinfo(Int.(ground_truth),group_labels))
            push!(nmi_score_history, mutualinfo(Int.(ground_truth),group_labels,normed=true))
        else
            push!(v_score_history, "no gt")
            push!(nmi_score_history, "no gt")
        end
        if use_verbose
            push!(liklihood_history,calculate_posterior(dp_model))
            println("Iteration: " * string(i) * " || Clusters count: " *
                string(cluster_count_history[end]) *
                " || Log posterior: " * string(liklihood_history[end]) *
                " || Vi score: " * string(v_score_history[end]) *
                " || NMI score: " * string(nmi_score_history[end]) *
                " || Iter Time:" * string(iter_time) *
                 " || Total time:" * string(sum(iter_count)))
        else
            push!(liklihood_history,1)
        end
        # if length(dp_model.group.local_clusters) > cur_parr_count
        #     cur_parr_count += max(20,length(dp_model.group.local_clusters))
        #     @sync for i in (nworkers()== 0 ? procs() : workers())
        #         @spawnat i set_parr_worker(dp_model.group.labels,cur_parr_count)
        #     end
        # end
        if i % model_save_interval == 0 && should_save_model
            println("Saving Model:")
            # save_time = time()
            @time save_model(dp_model,save_path, save_file_prefix, i, time() - start_time, model_params)
            # start_time += time() - save_time
        end
    end
    return dp_model, iter_count , nmi_score_history, liklihood_history, cluster_count_history
end


"""
    run_model_from_checkpoint(filename)

Run the model from a checkpoint created by it, `filename` is the path to the checkpoint.
Only to be run when using the advanced mode, note that the data must be in the same path as previously.

# Example:
```julia
julia> dp = run_model_from_checkpoint("checkpoint__50.jld2")
Loading Model:
  1.073261 seconds (2.27 M allocations: 113.221 MiB, 2.60% gc time)
Including params
Loading data:
  0.000881 seconds (10.02 k allocations: 378.313 KiB)
Creating model:
Node Leaders:
Dict{Any,Any}(2=>Any[2, 3])
Running model:
...
```
"""
function run_model_from_checkpoint(filename)
    println("Loading Model:")
    @time @load filename group hyperparams iter total_time global_params
    println("Including params")
    include(global_params)
    println("Loading data:")
    @time data = distribute((Float32.(load_data(data_path, prefix = data_prefix))))
    println("Creating model:")
    # @time begin
    group.labels = distribute(group.labels)
    group.labels_subcluster = distribute(group.labels_subcluster)
    dp_model = create_model_from_saved_data(group, data, hyperparams)
    # end
    global leader_dict = get_node_leaders_dict()
    println("Node Leaders:")
    println(leader_dict)
    @eval @everywhere global hard_clustering = $hard_clustering
    println("Running model:")
    return run_model(dp_model, iter+1, global_params, total_time)
end


function save_model(model, path, filename, iter, total_time, global_params)
    filename = path * filename * "_"*string(iter)*".jld2"
    hyperparams = model.model_hyperparams
    group = create_pts_less_group(model.group)
    @save filename group hyperparams iter total_time global_params
end


function calculate_posterior(model::dp_parallel_sampling)
    log_posterior = logabsgamma(model.model_hyperparams.α)[1] - logabsgamma(size(model.group.points,2)+model.model_hyperparams.α)[1]
    for cluster in model.group.local_clusters
        if cluster.cluster_params.cluster_params.suff_statistics.N == 0
            continue
        end
        log_posterior += log_marginal_likelihood(cluster.cluster_params.cluster_params.hyperparams,
            cluster.cluster_params.cluster_params.posterior_hyperparams,
            cluster.cluster_params.cluster_params.suff_statistics)
        log_posterior += log(model.model_hyperparams.α) + logabsgamma(cluster.cluster_params.cluster_params.suff_statistics.N)[1]
    end
    return log_posterior
end


function set_parr_worker(labels,cluster_count)
    global glob_parr = zeros(Float32,size(localpart(labels),1),cluster_count)
end

"""
    cluster_statistics(points,labels, clusters)

Provide avg statsitcs of probabiliy and likelihood for given points, labels and clusters

# Args and Kwargs
 - `points` a `DxN` array containing the data
 - `labels` points labels
 - `clusters` vector of clusters distributions


# Return values
avg_ll, avg_prob
 - `avg_ll` each cluster avg point ll
 - `avg_prob` each cluster avg point prob


# Example:
```julia
julia> dp = run_model_from_checkpoint("checkpoint__50.jld2")
Loading Model:
  1.073261 seconds (2.27 M allocations: 113.221 MiB, 2.60% gc time)
Including params
Loading data:
  0.000881 seconds (10.02 k allocations: 378.313 KiB)
Creating model:
Node Leaders:
Dict{Any,Any}(2=>Any[2, 3])
Running model:
...
```
"""
function cluster_statistics(points,labels, clusters)
    parr = zeros(Float32,length(labels), length(clusters))
    tic = time()
    for (k,cluster) in enumerate(clusters)
        log_likelihood!(reshape((@view parr[:,k]),:,1), points,cluster)
    end
    log_likelihood_array = copy(parr)
    log_likelihood_array[isnan.(log_likelihood_array)] .= -Inf #Numerical errors arent fun
    max_log_prob_arr = maximum(log_likelihood_array, dims = 2)
    log_likelihood_array .-= max_log_prob_arr
    map!(exp,log_likelihood_array,log_likelihood_array)
    # println("lsample log cat2" * string(log_likelihood_array))
    sum_prob_arr = sum(log_likelihood_array, dims =[2])
    log_likelihood_array ./=  sum_prob_arr
    avg_ll = zeros(length(clusters))
    avg_prob = zeros(length(clusters))
    for i=1:length(clusters)
        avg_ll[i] = sum(parr[labels .== i,i]) / sum(labels .== i)
        avg_prob[i] = sum(log_likelihood_array[labels .== i,i]) / sum(labels .== i)
    end
    return avg_ll, avg_prob
end
