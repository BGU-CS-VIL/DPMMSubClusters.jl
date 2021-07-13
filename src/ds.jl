abstract type distribution_hyper_params end
#Suff statistics must contain N which is the number of points associated with the cluster
abstract type sufficient_statistics end
abstract type distribution_sample end
import Base.copy

struct model_hyper_params
    distribution_hyper_params::distribution_hyper_params
    α::Float32
    total_dim::Int64
end

mutable struct cluster_parameters
    hyperparams::distribution_hyper_params
    distribution::distribution_sample
    suff_statistics::Vector{Tuple{sufficient_statistics,Number}}
    posterior_hyperparams::distribution_hyper_params
end

mutable struct splittable_cluster_params
    cluster_params::cluster_parameters
    cluster_params_l::cluster_parameters
    cluster_params_r::cluster_parameters
    lr_weights::AbstractArray{Float32, 1}
    splittable::Bool
    logsublikelihood_hist::AbstractArray{Float32,1}
end

mutable struct thin_cluster_params{T <: distribution_sample}
    cluster_dist::T
    l_dist::T
    r_dist::T
    lr_weights::AbstractArray{Float32, 1}
end


mutable struct thin_suff_stats
    cluster_suff::sufficient_statistics
    l_suff::sufficient_statistics
    r_suff::sufficient_statistics
end

mutable struct local_cluster
    cluster_params::splittable_cluster_params
    total_dim::Int64
    points_count::Number
    l_count::Int64
    r_count::Int64
end

mutable struct local_group
    model_hyperparams::model_hyper_params
    points::AbstractArray{Float32,2}
    labels::AbstractArray{Int64,1}
    labels_subcluster::AbstractArray{Int64,1}
    local_clusters::Vector{local_cluster}
    weights::Vector{Float32}
end

mutable struct pts_less_group
    model_hyperparams::model_hyper_params
    labels::AbstractArray{Int64,1}
    labels_subcluster::AbstractArray{Int64,1}
    local_clusters::Vector{local_cluster}
    weights::Vector{Float32}
end

mutable struct local_group_stats
    labels::AbstractArray{Int64,1}
    labels_subcluster::AbstractArray{Int64,1}
    local_clusters::Vector{local_cluster}
end


mutable struct dp_parallel_sampling
    model_hyperparams::model_hyper_params
    group::local_group
end

function copy_local_cluster(c::local_cluster)
    return deepcopy(c)
end


function create_pts_less_group(group::local_group)
    return pts_less_group(group.model_hyperparams , Array(group.labels), Array(group.labels_subcluster), group.local_clusters, group.weights)
end

function create_model_from_saved_data(group::pts_less_group, points::AbstractArray{Float32,2},model_hyperparams::model_hyper_params)
    group = local_group(group.model_hyperparams, points , group.labels, group.labels_subcluster, group.local_clusters, group.weights)
    return dp_parallel_sampling(model_hyperparams, group)
end
