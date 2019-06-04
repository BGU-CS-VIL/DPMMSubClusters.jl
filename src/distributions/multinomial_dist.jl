using LinearAlgebra
using Distributions
using PDMats


struct multinomial_dist <: distibution_sample
    α::AbstractArray{Float64,1}
end


function log_likelihood!(r::AbstractArray,x::AbstractArray, distibution_sample::multinomial_dist , group::Int64 = -1)
    r .= (distibution_sample.α' * x)[1,:]
end
