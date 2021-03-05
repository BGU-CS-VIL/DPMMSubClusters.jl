using LinearAlgebra
using Distributions

"""
    mv_gaussian(μ::AbstractArray{Float32,1}
        Σ::AbstractArray{Float32,2}
        invΣ::AbstractArray{Float32,2}
        logdetΣ::Float32
        invChol::UpperTriangular)
[Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
"""
struct mv_gaussian <: distribution_sample
    μ::AbstractArray{Float32,1}
    Σ::AbstractArray{Float32,2}
    invΣ::AbstractArray{Float32,2}
    logdetΣ::Float32
    invChol::UpperTriangular
end


function log_likelihood!(r::AbstractArray,x::AbstractMatrix, distribution_sample::mv_gaussian , group::Int64 = -1)
     z = x .- distribution_sample.μ
    dcolwise_dot!(r,z, distribution_sample.invΣ * z)
    r .= -((length(distribution_sample.Σ) * Float32(log(2π)) + distribution_sample.logdetΣ)/2) .-r/2
end
