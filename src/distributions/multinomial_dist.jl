using LinearAlgebra
using Distributions

"""
    multinomial_hyper(α::AbstractArray{Float32,1})
[Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
"""
struct multinomial_dist <: distribution_sample
    α::AbstractArray{Float32,1}
end


function log_likelihood!(r::AbstractArray,x::AbstractArray, distribution_sample::multinomial_dist , group::Int64 = -1)
    r .= (distribution_sample.α' * x)[1,:]
end
