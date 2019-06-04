function init_model()
    if random_seed != nothing
        @eval @everywhere Random.seed!($random_seed)
    end
    println("Loading and distributing data:")
    @time data = distribute(load_data(data_path, prefix = data_prefix))
    total_dim = size(data,2)
    println("α: "* string(α))
    model_hyperparams = model_hyper_params(hyper_params,α,total_dim)


    labels = distribute(rand(1:initial_clusters,(size(data,2))))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))
    # labels_subcluster = distribute(ones(size(data,2)))
    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float64[])
    return dp_parallel_sampling(model_hyperparams,group)
end

function init_model_from_data(all_data)
    if random_seed != nothing
        @eval @everywhere Random.seed!($random_seed)
    end
    println("Loading and distributing data:")
    @time data = distribute(all_data)
    total_dim = size(data,2)
    println("α: "* string(α))
    model_hyperparams = model_hyper_params(hyper_params,α,total_dim)


    labels = distribute(rand(1:initial_clusters,(size(data,2))))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))

    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float64[])
    return dp_parallel_sampling(model_hyperparams,group)
end


function init_first_clusters!(dp_model::dp_parallel_sampling, initial_cluster_count::Int64)
    for i=1:initial_cluster_count
        push!(dp_model.group.local_clusters, create_first_local_cluster(dp_model.group))
    end
    @sync update_suff_stats_posterior!(dp_model.group)
    sample_clusters!(dp_model.group,false)
    broadcast_cluster_params([create_thin_cluster_params(x) for x in dp_model.group.local_clusters],[1.0])
end

function dp_parallel(all_data::AbstractArray{Float64,2},local_hyper_params::distribution_hyper_params,iters::Int64, init_clusters::Int64 = 1,seed = nothing)
    iterations = iters
    random_seed = seed
    hyper_params = local_hyper_params
    initial_clusters = init_clusters
    dp_model = init_model_from_data(all_data)
    global leader_dict = get_node_leaders_dict()
    init_first_clusters!(dp_model, initial_clusters)
    println("Node Leaders:")
    println(leader_dict)
    @eval @everywhere global hard_clustering = $hard_clustering
    return run_model(dp_model, 1)
end


function dp_parallel(model_params::String)
    include(model_params)
    dp_model = init_model()
    global leader_dict = get_node_leaders_dict()
    init_first_clusters!(dp_model, initial_clusters)

    println("Node Leaders:")
    println(leader_dict)
    @eval @everywhere global hard_clustering = $hard_clustering
    return run_model(dp_model, 1 ,model_params)
end

function run_model(dp_model, first_iter, model_params="none", prev_time = 0)
    start_time= time()
    for i=first_iter:iterations
        # plot_group(dp_model.group)
        final = false
        no_more_splits = false
        if i >= iterations - argmax_sample_stop #We assume the cluters k has been setteled by now, and a low probability random split can do dmg
            final = true
        end
        if i >= iterations - split_stop
            no_more_splits = true
        end

        iter_time = time()

        group_step(dp_model.group, no_more_splits, final, i==1)
        println("Iteration: " * string(i) * " || Clusters count: " *
            string(length(dp_model.group.local_clusters)) *
            " || Log posterior: " * string(calculate_posterior(dp_model)) *
            " || Iter Time:" * string(time() - iter_time) *
             " || Total time:" * string(time() - start_time + prev_time))
        if i % model_save_interval == 0
            println("Saving Model:")
            save_time = time()
            @time save_model(dp_model,save_path, save_file_prefix, i, time() - start_time, model_params)
            start_time += time() - save_time
        end
    end
    return dp_model
end


#Running a saved model will results in added time, and might bug the random seed.
function run_model_from_checkpoint(filename)
    println("Loading Model:")
    @time @load filename group hyperparams iter total_time global_params
    println("Including params")
    include(global_params)
    println("Loading data:")
    @time data = distribute(load_data(data_path, prefix = data_prefix))
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
    log_posterior = log(model.model_hyperparams.α) - lgamma(size(model.group.points,2))
    for cluster in model.group.local_clusters
        if cluster.cluster_params.cluster_params.suff_statistics.N == 0
            continue
        end
        log_posterior += log_marginal_likelihood(cluster.cluster_params.cluster_params.hyperparams,
            cluster.cluster_params.cluster_params.posterior_hyperparams,
            cluster.cluster_params.cluster_params.suff_statistics)
        log_posterior += log(model.model_hyperparams.α) + lgamma(cluster.cluster_params.cluster_params.suff_statistics.N)
    end
    return log_posterior
end
