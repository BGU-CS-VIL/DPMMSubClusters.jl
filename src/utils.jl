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



# #Note that we expect the log_likelihood_array to be in rows (samples) x columns (clusters) , this is due to making it more efficent that way.
function sample_log_cat_array!(labels::AbstractArray{Int64,1}, log_likelihood_array::AbstractArray{Float32,2})
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


function create_sufficient_statistics(dist::distribution_hyper_params, pts::Array{Any,1})
    return create_sufficient_statistics(dist,dist, Array{Float32}(undef, 0, 0))
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


function get_node_leaders_dict()
    leader_dict = Dict()
    cur_leader = (nworkers()== 0 ? procs() : workers())[1]
    leader_dict[cur_leader] = []
    for i in (nworkers()== 0 ? procs() : workers())
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
    res::Float32 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += lgamma(x+(1-d)/2)
    end
    return res
end


function dcolwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    n = length(r)
    @simd for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        @simd for i = 1:size(a, 1)
            @fastmath @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
end
