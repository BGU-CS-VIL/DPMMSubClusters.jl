
#Load some packages and add workers
using Images
using Distributed

using LinearAlgebra
using Statistics

#Load the package on all workers
addprocs(4)
@everywhere using DPMMSubClusters

#Load image, taken from SINTEL http://sintel.is.tue.mpg.de/
img = load("frame_0001.png")

#Change to channel view
img_channels = channelview(img)
z,x,y = size(img_channels)

#Create input
input_arr = zeros(5,x*y)
for i=1:x
    for j=1:y
        input_arr[:,(i-1)*y+j]=[img_channels[1,i,j],img_channels[2,i,j],img_channels[3,i,j],i,j]
    end
end

#Create HyperParams
#The rgb,xy multiplier allows us to both play with the weight of the xy/rgb
rgb_prior_multiplier = 1
xy_prior_multiplier = 0.1

data_cov = cov(input_arr')
data_cov[4:5,1:3] .= 0
data_cov[1:3,4:5] .= 0

data_cov[1:3,1:3] .*= rgb_prior_multiplier
data_cov[4:5,4:5] .*= xy_prior_multiplier

data_mean = mean(input_arr,dims = 2)[:]

hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
           data_mean,
           8,
           data_cov)

#Run the model
labels,clusters,weights = DPMMSubClusters.fit(input_arr,hyper_params,50000.0,iters = 200, verbose = true)

#Get the cluster color means
color_means = [x.μ[1:3] for x in clusters]


segemnated_image = zeros(3,x,y)
for i=1:x
    for j=1:y
        segemnated_image[:,i,j] = color_means[labels[(i-1)*y+j]]
    end
end
segemnated_image = colorview(RGB,segemnated_image)
