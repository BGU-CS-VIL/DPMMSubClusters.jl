using LinearAlgebra
using Distributions

"""
    mv_gaussian(μ::AbstractArray{Float64,1}
        Σ::AbstractArray{Float64,2}
        invΣ::AbstractArray{Float64,2}
        logdetΣ::Float64
        invChol::UpperTriangular)
[Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
"""
struct mv_gaussian <: distibution_sample
    μ::AbstractArray{Float64,1}
    Σ::AbstractArray{Float64,2}
    invΣ::AbstractArray{Float64,2}
    logdetΣ::Float64
    invChol::UpperTriangular
end


function log_likelihood!(r::AbstractArray,x::AbstractMatrix, distibution_sample::mv_gaussian , group::Int64 = -1)
     z = x .- distibution_sample.μ
    dcolwise_dot!(r,z, distibution_sample.invΣ * z)
    r .= -((length(distibution_sample.Σ) * Float64(log(2π)) + distibution_sample.logdetΣ)/2) .-r/2
end
