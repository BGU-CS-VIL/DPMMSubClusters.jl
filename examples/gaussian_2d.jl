@everywhere using DPMMSubClusters
using Plots
using LinearAlgebra



function plot_dp_2d(dp_model)
    group = dp_model.group
    labels = Array(group.labels)
    pts = Array(group.points)
    plt=Plots.plot()
    Plots.plot!(pts[1,:],pts[2,:], seriestype=:scatter, color = labels, markersize = 3, markerstrokewidth = 0.5)
    display(plt)
    return plt
end



x,labels,clusters = generate_gaussian_data(10^5,2,6)

hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
           zeros(2),
           4,
           Matrix{Float64}(I, 2, 2)*1)

dp = dp_parallel(x,hyper_params, 100, 1)

plot_dp_2d(dp)
