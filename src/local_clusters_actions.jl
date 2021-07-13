function create_first_local_cluster(group::local_group)
    suff = create_sufficient_statistics(group.model_hyperparams.distribution_hyper_params, [])
    post = group.model_hyperparams.distribution_hyper_params
    dist = sample_distribution(post)
    cp = cluster_parameters(group.model_hyperparams.distribution_hyper_params, dist, [(suff,0)], post)
    cpl = deepcopy(cp)
    cpr = deepcopy(cp)
    splittable = splittable_cluster_params(cp,cpl,cpr,[0.5,0.5], false,ones(burnout_period+5)*-Inf)
    cp.suff_statistics[1][1].N = size(group.points,2)
    cpl.suff_statistics[1][1].N = sum(group.labels_subcluster .== 1)
    cpl.suff_statistics[1][1].N = sum(group.labels_subcluster .== 2)
    cluster = local_cluster(splittable, group.model_hyperparams.total_dim,
        cp.suff_statistics[1][1].N,
        cpl.suff_statistics[1][1].N,
        cpl.suff_statistics[1][1].N)
    @sync for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i split_first_cluster_worker!(group)
    end
    return cluster
end


function create_outlier_local_cluster(group::local_group,outlier_params)
    suff = create_sufficient_statistics(outlier_params,outlier_params, Array(group.points))
    post = calc_posterior(outlier_params,suff)
    dist = sample_distribution(post)
    cp = cluster_parameters(outlier_params, dist, [(suff,0)], post)
    cpl = deepcopy(cp)
    cpr = deepcopy(cp)
    splittable = splittable_cluster_params(cp,cpl,cpr,[0.5,0.5], false,ones(burnout_period+5)*-Inf)
    cp.suff_statistics[1][1].N = size(group.points,2)
    cpl.suff_statistics[1][1].N = sum(group.labels_subcluster .== 1)
    cpl.suff_statistics[1][1].N = sum(group.labels_subcluster .== 2)
    cluster = local_cluster(splittable, group.model_hyperparams.total_dim,
        cp.suff_statistics[1][1].N,
        cpl.suff_statistics[1][1].N,
        cpl.suff_statistics[1][1].N)
    # @sync for i in (nworkers()== 0 ? procs() : workers())
    #     @spawnat i split_first_cluster_worker!(group)
    # end
    return cluster
end


function sample_sub_clusters!(group::local_group)
    for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i sample_sub_clusters_worker!(group.points, group.labels, group.labels_subcluster)
    end
end

function sample_sub_clusters_worker!(group_points, group_labels, group_labels_subcluster)

    pts = localpart(group_points)
    labels = localpart(group_labels)
    sub_labels = localpart(group_labels_subcluster)

    for (i,v) in enumerate(clusters_vector)
        create_subclusters_labels!((@view sub_labels[labels .== i]),
            (@view pts[:,labels .== i]), v)
    end

end

function create_subclusters_labels!(labels::AbstractArray{Int64,1},
        points::AbstractArray{Float32,2},
        cluster_params::thin_cluster_params)
    if size(labels,1) == 0
        return
    end
    parr = zeros(Float32,length(labels), 2)
    log_likelihood!((@view parr[:,1]),points,cluster_params.l_dist)
    log_likelihood!((@view parr[:,2]),points,cluster_params.r_dist)
    parr[:,1] .+= log(cluster_params.lr_weights[1])
    parr[:,2] .+= log(cluster_params.lr_weights[2])
    sample_log_cat_array!(labels,parr)
end


function sample_labels!(group::local_group, final::Bool, no_more_splits::Bool)
    sample_labels!(group.labels, group.points, final, no_more_splits)
end

function sample_labels!(labels::AbstractArray{Int64,1},
        points::AbstractArray{Float32,2},
        final::Bool,
        no_more_splits::Bool)
    for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i sample_labels_worker!(labels,points,final, no_more_splits)
    end
end


function sample_labels_worker!(labels::AbstractArray{Int64,1},
        points::AbstractArray{Float32,2},
        final::Bool,
        no_more_splits::Bool)
    indices = localindices(points)[2]
    lbls = localpart(labels)
    pts = localpart(points)
    log_weights = log.(clusters_weights)
    parr = zeros(Float32,length(indices), length(clusters_vector))
    tic = time()
    for (k,cluster) in enumerate(clusters_vector)
        log_likelihood!(reshape((@view parr[:,k]),:,1), pts,cluster.cluster_dist)
    end
    for (k,v) in enumerate(clusters_weights)
        parr[:,k] .+= log(v)
    end

    if final
        lbls .= mapslices(argmax, parr, dims= [2])[:]
    else
        sample_log_cat_array!(lbls,parr)
    end
end


function update_splittable_cluster_params!(splittable_cluser::splittable_cluster_params)
    cpl = splittable_cluser.cluster_params_l
    cpr = splittable_cluser.cluster_params_r
    cp = splittable_cluser.cluster_params

    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
    cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)


end

function create_suff_stats_dict_worker(group_pts, group_labels, group_sublabels, hyper_params, indices)
    suff_stats_dict = Dict()
    if indices == nothing
        indices = collect(1:length(clusters_vector))
    end

    points = localpart(group_pts)
    labels = localpart(group_labels)
    sublabels = localpart(group_sublabels)
    for index in indices
        pts = @view points[:, labels .== index]
        sub_labels = @view sublabels[labels .== index]


        cpl_suff = create_sufficient_statistics(hyper_params,hyper_params, @view pts[:,sub_labels .== 1])
        cpr_suff = create_sufficient_statistics(hyper_params,hyper_params, @view pts[:,sub_labels .== 2])
        cp_suff = create_sufficient_statistics(hyper_params,hyper_params, pts)
        suff_stats_dict[index] = thin_suff_stats(cp_suff,cpl_suff,cpr_suff)
    end
    return suff_stats_dict
end

function create_suff_stats_dict_node_leader(group_pts, group_labels, group_sublabels, hyper_params, proc_ids, indices)
    leader_suff_dict = Dict()
    workers_suff_dict = Dict()
    if indices == nothing
        indices = collect(1:length(clusters_vector))
    end
    for i in proc_ids
        workers_suff_dict[i] = remotecall(create_suff_stats_dict_worker,i,group_pts,
            group_labels,
            group_sublabels,
            hyper_params,
            indices)
    end
    suff_stats_vectors = [[] for i=1:length(indices)]
    cluster_to_index = Dict([indices[i]=>i for i=1:length(indices)])
    workers_suff_dict_fetched = Dict([k=>fetch(v) for (k,v) in workers_suff_dict])
    for (k,v) in workers_suff_dict_fetched
        for (cluster, suff) in v
            push!(suff_stats_vectors[cluster_to_index[cluster]], suff)
        end
    end
    for (k,v) in enumerate(suff_stats_vectors)
        if length(v) > 0
            cp_suff = reduce(aggregate_suff_stats, [x.cluster_suff for x in v])
            cpl_suff = reduce(aggregate_suff_stats, [x.l_suff for x in v])
            cpr_suff = reduce(aggregate_suff_stats, [x.r_suff for x in v])
            cluster_index = indices[k]
            leader_suff_dict[cluster_index] = thin_suff_stats(cp_suff,cpl_suff,cpr_suff)
        end
    end

    return leader_suff_dict
end


function update_suff_stats_posterior!(group::local_group,indices = nothing, use_leader::Bool = true)
    workers_suff_dict = Dict()
    global ϵ
    if indices == nothing
        indices = collect(1:length(group.local_clusters))
    end
    if use_leader
        for i in collect(keys(leader_dict))
            workers_suff_dict[i] = remotecall(create_suff_stats_dict_node_leader, i ,group.points,
                group.labels,
                group.labels_subcluster,
                group.model_hyperparams.distribution_hyper_params,
                leader_dict[i],
                indices)
        end
    else
        for i in (nworkers()== 0 ? procs() : workers())
            workers_suff_dict[i] = @spawnat i create_suff_stats_dict_worker(group.points,
                group.labels,
                group.labels_subcluster,
                group.model_hyperparams.distribution_hyper_params,
                indices)
        end
    end
    suff_stats_vectors = [[] for i=1:length(indices)]
    cluster_to_index = Dict([indices[i]=>i for i=1:length(indices)])
    workers_suff_dict_fetched = Dict([k=>fetch(v) for (k,v) in workers_suff_dict])
    for (k,v) in workers_suff_dict_fetched
        for (cluster, suff) in v
            push!(suff_stats_vectors[cluster_to_index[cluster]], suff)
        end
    end
    for (index,v) in enumerate(indices)
        # if outlier_mod > 0 && v == 1
        #     continue
        # end
        if length(suff_stats_vectors[index]) == 0
            continue
        end
        cluster = group.local_clusters[v]
        cp = cluster.cluster_params
        suff_stat = reduce(aggregate_suff_stats, [x.cluster_suff for x in suff_stats_vectors[index]])
        suff_stat_l = reduce(aggregate_suff_stats, [x.l_suff for x in suff_stats_vectors[index]])
        suff_stat_r = reduce(aggregate_suff_stats, [x.r_suff for x in suff_stats_vectors[index]])
        if cp.cluster_params.suff_statistics[end][2] == global_time
            cp.cluster_params.suff_statistics[end] = (suff_stat,global_time)
            cp.cluster_params_l.suff_statistics[end] = (suff_stat_l,global_time)
            cp.cluster_params_r.suff_statistics[end] = (suff_stat_r,global_time)
        else
            push!(cp.cluster_params.suff_statistics,(suff_stat,global_time))
            push!(cp.cluster_params_l.suff_statistics,(suff_stat_l,global_time))
            push!(cp.cluster_params_r.suff_statistics,(suff_stat_r,global_time))
        end
        deleteat!(cp.cluster_params.suff_statistics, findall(x->post_kernel(x[2],global_time)<=ϵ, cp.cluster_params.suff_statistics))
        deleteat!(cp.cluster_params_l.suff_statistics, findall(x->post_kernel(x[2],global_time)<=ϵ, cp.cluster_params_l.suff_statistics))
        deleteat!(cp.cluster_params_r.suff_statistics, findall(x->post_kernel(x[2],global_time)<=ϵ, cp.cluster_params_r.suff_statistics))
        cluster.points_count = sum([post_kernel(x[2],global_time)*x[1].N for x in cp.cluster_params.suff_statistics])
        update_splittable_cluster_params!(cluster.cluster_params)
    end


end


function split_first_cluster_worker!(group::local_group)
    sub_labels = localpart(group.labels_subcluster)
    pts = localpart(group.points)
    sub_labels .= rand(1:2,length(sub_labels))
end



function split_cluster_local_worker!(labels, sub_labels, points,indices::Vector{Int64}, new_indices::Vector{Int64})
    labels = localpart(labels)
    sub_labels = localpart(sub_labels)
    pts = localpart(points)
    for (i,index) in enumerate(indices)
        cluster_sub_labels = @view sub_labels[labels .== index]
        cluster_labels = @view labels[labels .== index]
        cluster_points = @view pts[:,labels .== index]
        cluster_labels[cluster_sub_labels .== 2] .= new_indices[i]

        cluster_sub_labels .= rand(1:2,length(cluster_sub_labels))

    end
end

function split_cluster_local!(group::local_group, cluster::local_cluster, index::Int64, new_index::Int64)

    l_split = copy_local_cluster(cluster)
    l_split.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, group.model_hyperparams.α)
    cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, group.model_hyperparams.α)    
    l_split.points_count = sum([post_kernel(x[2],global_time)*x[1].N for x in l_split.cluster_params.cluster_params.suff_statistics])
    cluster.points_count = sum([post_kernel(x[2],global_time)*x[1].N for x in cluster.cluster_params.cluster_params.suff_statistics])
    group.local_clusters[new_index] = l_split
end

function merge_clusters_worker!(group::local_group,indices::Vector{Int64}, new_indices::Vector{Int64})
    labels = localpart(group.labels)
    sub_labels = localpart(group.labels_subcluster)

    for (i,index) in enumerate(indices)
        cluster_sub_labels = @view localpart(group.labels_subcluster)[labels .== index]
        cluster_sub_labels .= 1
        cluster_sub_labels = @view localpart(group.labels_subcluster)[labels .== new_indices[i]]
        cluster_sub_labels .= 2
        labels[labels .== new_indices[i]] .= index
    end
end



function merge_clusters!(group::local_group,index_l::Int64, index_r::Int64)
    new_splittable_cluster = merge_clusters_to_splittable(group.local_clusters[index_l].cluster_params.cluster_params, group.local_clusters[index_r].cluster_params.cluster_params, group.model_hyperparams.α)
    group.local_clusters[index_l].cluster_params = new_splittable_cluster
    group.local_clusters[index_l].points_count += group.local_clusters[index_r].points_count
    group.local_clusters[index_r].points_count = 0
    group.local_clusters[index_r].cluster_params.cluster_params.suff_statistics.N = 0
    group.local_clusters[index_r].cluster_params.splittable = false
end


function should_split_local!(should_split::AbstractArray{Float32,1},
        cluster_params::splittable_cluster_params, α::Float32, final::Bool)
    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    sum([post_kernel(x[2],global_time)*x[1].N for x in cpl.suff_statistics])
    if final || sum([post_kernel(x[2],global_time)*x[1].N for x in cpl.suff_statistics]) == 0 ||sum([post_kernel(x[2],global_time)*x[1].N for x in cpr.suff_statistics]) == 0
        should_split .= 0
        return
    end
    post = calc_posterior(cp.hyperparams,cp.suff_statistics)
    lpost = calc_posterior(cp.hyperparams,cpl.suff_statistics)
    rpost = calc_posterior(cp.hyperparams,cpr.suff_statistics)


    log_likihood_l = log_marginal_likelihood(cpl.hyperparams,lpost, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams,rpost, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, post, cp.suff_statistics)

    log_HR = log(α) +
        logabsgamma(sum([post_kernel(x[2],global_time)*x[1].N for x in cpl.suff_statistics]))[1] + log_likihood_l +
        logabsgamma(sum([post_kernel(x[2],global_time)*x[1].N for x in cpr.suff_statistics]))[1] + log_likihood_r -
        (logabsgamma(sum([post_kernel(x[2],global_time)*x[1].N for x in cp.suff_statistics]))[1] + log_likihood)
    if log_HR > log(rand())
        should_split .= 1
    end
end

function check_and_split!(group::local_group, final::Bool)
    split_arr= zeros(Float32,length(group.local_clusters))
    for (index,cluster) in enumerate(group.local_clusters)
        if outlier_mod > 0 && index == 1
            continue
        end
        
        if cluster.cluster_params.splittable == true && sum([post_kernel(x[2],global_time)*x[1].N for x in cluster.cluster_params.cluster_params.suff_statistics]) > 1
            should_split_local!((@view split_arr[index,:]), cluster.cluster_params,
                group.model_hyperparams.α,final)
        end
    end
    new_index = length(group.local_clusters) + 1
    indices = Vector{Int64}()
    new_indices = Vector{Int64}()
    resize!(group.local_clusters,Int64(length(group.local_clusters) + sum(split_arr)))
    for i=1:length(split_arr)
        if split_arr[i] == 1

            push!(indices, i)
            push!(new_indices, new_index)
            split_cluster_local!(group, group.local_clusters[i],i,new_index)
            new_index += 1
        end
    end
    all_indices = vcat(indices,new_indices)
    if length(indices) > 0
        for i in (nworkers()== 0 ? procs() : workers())
            @spawnat i split_cluster_local_worker!(group.labels,group.labels_subcluster,group.points,indices,new_indices)
        end
    end
    return all_indices
end


function check_and_merge!(group::local_group, final::Bool)
    mergable = zeros(Float32,1)
    indices = Vector{Int64}()
    new_indices = Vector{Int64}()
    for i=1:length(group.local_clusters)
        if outlier_mod > 0 && i == 1
            continue
        end
        sum([post_kernel(x[2],global_time)*x[1].N for x in group.local_clusters[i].cluster_params.cluster_params.suff_statistics])
        for j=i+1:length(group.local_clusters)
            if  (group.local_clusters[i].cluster_params.splittable == true &&
                    group.local_clusters[j].cluster_params.splittable == true &&
                    sum([post_kernel(x[2],global_time)*x[1].N for x in group.local_clusters[i].cluster_params.cluster_params.suff_statistics]) > 0 &&
                    sum([post_kernel(x[2],global_time)*x[1].N for x in group.local_clusters[j].cluster_params.cluster_params.suff_statistics]) > 0)
                should_merge!(mergable, group.local_clusters[i].cluster_params.cluster_params,
                    group.local_clusters[j].cluster_params.cluster_params, group.model_hyperparams.α, final)
            end
            if mergable[1] == 1
                merge_clusters!(group, i, j)
                push!(indices, i)
                push!(new_indices, j)
            end
            mergable[1] = 0
        end
    end
    for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i merge_clusters_worker!(group,indices,new_indices)
    end
    return indices
end



function sample_clusters!(group::local_group, first::Bool)
    points_count = Vector{Float32}()
    local_workers = procs(1)[2:end]
    cluster_params_futures = Dict()
    for (i,cluster) in enumerate(group.local_clusters)
        cluster_params_futures[i] = sample_cluster_params(cluster.cluster_params, group.model_hyperparams.α,first)
    end
    for (i,cluster) in enumerate(group.local_clusters)
        if outlier_mod > 0 && i == 1
            continue
        end
        cluster.cluster_params = fetch(cluster_params_futures[i])        
        cluster.points_count = sum([post_kernel(x[2],global_time)*x[1].N for x in cluster.cluster_params.cluster_params.suff_statistics])
        push!(points_count, cluster.points_count)
    end
    push!(points_count, group.model_hyperparams.α)
    group.weights = rand(Dirichlet(Float64.(points_count)))[1:end-1] .* (1 - outlier_mod)
    if outlier_mod > 0
        group.weights = vcat([outlier_mod],group.weights)
    end
end

function create_thin_cluster_params(cluster::local_cluster)
    return thin_cluster_params(cluster.cluster_params.cluster_params.distribution,
        cluster.cluster_params.cluster_params_l.distribution,
        cluster.cluster_params.cluster_params_r.distribution,
        cluster.cluster_params.lr_weights)
end

function remove_empty_clusters_worker!(labels, pts_count)
    labels = localpart(labels)
    removed = 0
    for (index, count) in enumerate(pts_count)
        if count == 0
            labels[labels .> index - removed] .-= 1
            removed += 1
        end
    end
end

function remove_empty_clusters!(group::local_group)
    new_vec = Vector{local_cluster}()
    removed = 0
    pts_count = Vector{Number}()
    for (index,cluster) in enumerate(group.local_clusters)
        push!(pts_count, cluster.points_count)
        if cluster.points_count >= 1 || (outlier_mod > 0 && index == 1) || (outlier_mod > 0 && index == 2 && length(group.local_clusters) == 2)
            push!(new_vec,cluster)
        end
    end
    @sync for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i remove_empty_clusters_worker!(group.labels, pts_count)
    end
    group.local_clusters = new_vec
end


function rand_subclusters_labels!(labels::AbstractArray{Int64,1})    #lr_arr = create_array(zeros(Float32,length(labels), 2))
    if size(labels,1) == 0
        return
    end
    labels .= rand(1:2,size(labels,1))
end

function reset_bad_clusters_worker!(indices::Vector{Int64}, group_points::AbstractArray{Float32,2},group_labels::AbstractArray{Int64,1}, group_labels_subcluster::AbstractArray{Int64,1})
    labels = localpart(group_labels)
    sub_labels = localpart(group_labels_subcluster)
    pts = localpart(group_points)
    for i in indices
        rand_subclusters_labels!((@view sub_labels[labels .== i]))
    end
end

function reset_splitted_clusters!(group::local_group, bad_clusters::Vector{Int64})

    for i in bad_clusters
        group.local_clusters[i].cluster_params.logsublikelihood_hist = ones(burnout_period+5)*-Inf
    end
    for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i reset_bad_clusters_worker!(bad_clusters,group.points, group.labels, group.labels_subcluster)
    end
    update_suff_stats_posterior!(group,bad_clusters)
end

function reset_bad_clusters!(group::local_group)
    bad_clusters = Vector{Int64}()
    for (i,c) in enumerate(group.local_clusters)
        cl = c.cluster_params.cluster_params_l
        cr = c.cluster_params.cluster_params_r
        sum([post_kernel(x[2],global_time)*x[1].N for x in cr.suff_statistics])
        if sum([post_kernel(x[2],global_time)*x[1].N for x in cl.suff_statistics]) == 0 || sum([post_kernel(x[2],global_time)*x[1].N for x in cr.suff_statistics]) == 0
            push!(bad_clusters,i)
            c.cluster_params.logsublikelihood_hist = ones(burnout_period+5)*-Inf
            c.cluster_params.splittable = false
        end
    end
    @sync for i in (nworkers()== 0 ? procs() : workers())
        @spawnat i reset_bad_clusters_worker!(bad_clusters,group.points, group.labels, group.labels_subcluster)
    end
    update_suff_stats_posterior!(group,bad_clusters)
end

function broadcast_cluster_params(params_vector, weights_vector)
    @sync for (k,v) in leader_dict
        @spawnat k broadcast_to_node(params_vector, weights_vector,v)
    end
end


#Below code is to prevent some bug with deseraliztion
function broadcast_to_node(params_vector, weights_vector, proc_ids)
    refs = Dict()
    @sync for i in proc_ids
        refs[i] = @spawnat i set_global_data(params_vector, weights_vector)
    end
    alltrue = false
    responses = Dict([k =>fetch(v) for (k,v) in refs])
    for (k,v) in responses
        while v == false
            v = remotecall_fetch(set_global_data, k, params_vector, weights_vector)
        end
    end
end

function set_global_data(params_vector, weights_vector)
    succ = true
    try
        global clusters_vector = params_vector
        global clusters_weights = weights_vector
    catch
        succ = false
    end
    return succ
end

function group_step(group::local_group, no_more_splits::Bool, final::Bool,first::Bool,cur_time::Number)
    sample_clusters!(group,false)
    broadcast_cluster_params([create_thin_cluster_params(x) for x in group.local_clusters],group.weights)
    global global_time = cur_time
    sample_labels!(group, (hard_clustering ? true : final), no_more_splits)
    sample_sub_clusters!(group)
    update_suff_stats_posterior!(group)
    reset_bad_clusters!(group)
    if no_more_splits == false
        indices = []
        indices = check_and_split!(group, final)
        update_suff_stats_posterior!(group, indices)
        check_and_merge!(group, final)
    end
    remove_empty_clusters!(group)
    return
end
