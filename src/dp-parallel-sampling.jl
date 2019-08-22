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
    labels = distribute(rand(1:initial_clusters,(size(data,2))))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))
    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float32[])
    return dp_parallel_sampling(model_hyperparams,group)
end

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
    labels = distribute(rand(1:initial_clusters,(size(data,2))))
    labels_subcluster = distribute(rand(1:2,(size(data,2))))
    group = local_group(model_hyperparams,data,labels,labels_subcluster,local_cluster[],Float32[])
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

function dp_parallel(all_data::AbstractArray{Float32,2},
        local_hyper_params::distribution_hyper_params,
        α_param::Float32,
         iters::Int64 = 100,
         init_clusters::Int64 = 1,
         seed = nothing,
         verbose = true,
         save_model = false,
         burnout = 15,
         gt = nothing)
    global iterations = iters
    global random_seed = seed
    global hyper_params = local_hyper_params
    global initial_clusters = init_clusters
    global α = α_param
    global use_verbose = verbose
    global should_save_model = save_model
    global burnout_period = burnout
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

function fit(all_data::AbstractArray{Float32,2},local_hyper_params::distribution_hyper_params,α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false, burnout = 20, gt = nothing)
        dp_model,iter_count , nmi_score_history, liklihood_history, cluster_count_history = dp_parallel(all_data, local_hyper_params,α_param,iters,init_clusters,seed,verbose,save_model,burnout,gt)
        return Array(dp_model.group.labels), [x.cluster_params.cluster_params.distribution for x in dp_model.group.local_clusters], dp_model.group.weights,iter_count , nmi_score_history, liklihood_history, cluster_count_history
end


function fit(all_data::AbstractArray{Float32,2},α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false,burnout = 20, gt = nothing)
    data_dim = size(all_data,1)
    cov_mat = Matrix{Float32}(I, data_dim, data_dim)
    local_hyper_params = niw_hyperparams(1,zeros(Float32,data_dim),data_dim+3,cov_mat)
    dp_model,iter_count , nmi_score_history, liklihood_history, cluster_count_history = dp_parallel(all_data, local_hyper_params,α_param,iters,init_clusters,seed,verbose,save_model,burnout,gt)
    return Array(dp_model.group.labels), [x.cluster_params.cluster_params.distribution for x in dp_model.group.local_clusters], dp_model.group.weights,iter_count , nmi_score_history, liklihood_history, cluster_count_history
end

fit(all_data::AbstractArray, α_param;
        iters = 100, init_clusters = 1,
        seed = nothing, verbose = true,
        save_model = false,burnout = 20, gt = nothing) =
    fit(Float32.(all_data),Float32(α_param),iters = Int64(iters),
        init_clusters=Int64(init_clusters), seed = seed, verbose = verbose,
        save_model = save_model, burnout = burnout, gt = gt)


fit(all_data::AbstractArray,local_hyper_params::distribution_hyper_params,α_param;
        iters = 100, init_clusters::Number = 1,
        seed = nothing, verbose = true,
        save_model = false,burnout = 20, gt = nothing) =
    fit(Float32.(all_data),local_hyper_params,Float32(α_param),iters = Int64(iters),
        init_clusters=Int64(init_clusters), seed = seed, verbose = verbose,
        save_model = save_model, burnout = burnout, gt = gt)



function dp_parallel(model_params::String; verbose = true, save_model = true,burnout = 5, gt = nothing)
    include(model_params)
    global use_verbose = verbose
    dp_model = init_model()
    global leader_dict = get_node_leaders_dict()
    global should_save_model = save_model
    global ground_truth = gt
    global burnout_period = burnout
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
        if i >= iterations - split_stop
            no_more_splits = true
        end

        prev_time = time()
        group_step(dp_model.group, no_more_splits, final, i==1)
        iter_time = time() - prev_time
        push!(iter_count,iter_time)
        push!(liklihood_history,calculate_posterior(dp_model))
        push!(cluster_count_history,length(dp_model.group.local_clusters))
        group_labels = Array(dp_model.group.labels)
        if ground_truth != nothing
            push!(v_score_history, varinfo(Int(maximum(ground_truth)), Int.(ground_truth), length(unique(group_labels)),group_labels))
            push!(nmi_score_history, mutualinfo(Int.(ground_truth),group_labels)[2])
        else
            push!(v_score_history, "no gt")
            push!(nmi_score_history, "no gt")
        end
        if use_verbose
            println("Iteration: " * string(i) * " || Clusters count: " *
                string(cluster_count_history[end]) *
                " || Log posterior: " * string(liklihood_history[end]) *
                " || Vi score: " * string(v_score_history[end]) *
                " || NMI score: " * string(nmi_score_history[end]) *
                " || Iter Time:" * string(iter_time) *
                 " || Total time:" * string(sum(iter_count)))
        end
        if length(dp_model.group.local_clusters) > cur_parr_count
            cur_parr_count += max(20,length(dp_model.group.local_clusters))
            @sync for i in (nworkers()== 0 ? procs() : workers())
                @spawnat i set_parr_worker(dp_model.group.labels,cur_parr_count)
            end
        end
        if i % model_save_interval == 0 && should_save_model
            println("Saving Model:")
            # save_time = time()
            @time save_model(dp_model,save_path, save_file_prefix, i, time() - start_time, model_params)
            # start_time += time() - save_time
        end
    end
    return dp_model, iter_count , nmi_score_history, liklihood_history, cluster_count_history
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


function set_parr_worker(labels,cluster_count)
    global glob_parr = zeros(Float32,size(localpart(labels),1),cluster_count)
end
