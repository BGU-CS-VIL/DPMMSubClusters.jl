using NPZ


# We expects the data to be in npy format, return a dict of {group: items}, each file is a different group
function load_data(path::String; prefix::String="", swapDimension::Bool = true)
    groups_dict = Dict()
    arr = npzread(path * prefix * ".npy")
    for (index, value) in enumerate(arr)
        if isnan(value)
            arr[index] = 0.0
        end
    end
    return swapDimension ? transpose(arr) : arr
end


function dcolwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    n = length(r)
    for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        for i = 1:size(a, 1)
            @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
end

function dcolwise_dot(a::AbstractMatrix, b::AbstractMatrix)
    n = size(a,2)
    r = zeros(n)
    for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        for i = 1:size(a, 1)
            @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
    return r
end


# #Note that we expect the log_likelihood_array to be in rows (samples) x columns (clusters) , this is due to making it more efficent that way.
function sample_log_cat_array!(labels::AbstractArray{Int64,1}, log_likelihood_array::AbstractArray{Float64,2})
    # println("lsample log cat" * string(log_likelihood_array))
    max_log_prob_arr = maximum(log_likelihood_array, dims = 2)
    log_likelihood_array .-= max_log_prob_arr
    map!(exp,log_likelihood_array,log_likelihood_array)
    # println("lsample log cat2" * string(log_likelihood_array))
    sum_prob_arr = sum(log_likelihood_array, dims =[2])
    log_likelihood_array ./=  sum_prob_arr
    for i=1:length(labels)
        labels[i] = sample(1:size(log_likelihood_array,2), ProbabilityWeights(log_likelihood_array[i,:]))
    end
end

function sample_log_cat_array!(labels::AbstractArray{Int64,1}, log_likelihood_array::AbstractArray{Float64,2}, weights_vector::AbstractArray{Float64,1})
    # println("lsample log cat" * string(log_likelihood_array))
    max_log_prob_arr = maximum(log_likelihood_array, dims = 2)
    log_likelihood_array .-= max_log_prob_arr
    map!(exp,log_likelihood_array,log_likelihood_array)
    # println("lsample log cat2" * string(log_likelihood_array))
    sum_prob_arr = sum(log_likelihood_array, dims =[2])
    log_likelihood_array ./=  sum_prob_arr
    for i=1:length(weights_vector)
        log_likelihood_array[:,i] .*= weights_vector[i]
    end
    for i=1:length(labels)
        old_label = labels[i]
        labels[i] = sample(1:size(log_likelihood_array,2), ProbabilityWeights(log_likelihood_array[i,:]))
        new_label = labels[i]
        if old_label == 7 && new_label == 10
            println(log_likelihood_array[i,[7,10]])
        end
    end
end


function sample_log_cat(logcat_array::AbstractArray{Float64, 1})
    max_logprob::Float64 = maximum(logcat_array)
    for i=1:length(logcat_array)
        logcat_array[i] = exp(logcat_array[i]-max_logprob)
    end
    sum_logprob::Float64 = sum(logcat_array)
    i::Int64 = 1
    c::Float64 = logcat_array[1]
    u::Float64 = rand()*sum(logcat_array)
    while c < u && i < length(logcat_array)
        c += logcat_array[i += 1]
    end
    return i
end


function create_sufficient_statistics(dist::distribution_hyper_params, pts::Array{Any,1})
    return create_sufficient_statistics(dist,dist, Array{Float64}(undef, 0, 0))
end


function get_labels_histogram(labels)
    hist_dict = Dict()
    for v in labels
        if haskey(hist_dict,v) == false
            hist_dict[v] = 0
        end
        hist_dict[v] += 1
    end
    return sort(collect(hist_dict), by=x->x[1])
end

function create_global_labels(group::local_group)
    clusters_dict = Dict()
    for (i,v) in enumerate(group.local_clusters)
        clusters_dict[i] = v.globalCluster
    end
    return [clusters_dict[i] for i in group.labels]
end

function get_node_leaders_dict()
    leader_dict = Dict()
    cur_leader = 2
    leader_dict[cur_leader] = []
    for i in workers()
        if i in procs(cur_leader)
            push!(leader_dict[cur_leader], i)
        else
            cur_leader = i
            leader_dict[cur_leader] = [i]
        end
    end
    return leader_dict
end

function log_multivariate_gamma(x::Number, D::Number)
    res::Float64 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += lgamma(x+(1-d)/2)
    end
    return res
end
