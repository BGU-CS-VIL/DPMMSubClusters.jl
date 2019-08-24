# Getting Started

## Installation
The package is available through Julia package repository.
Install by either:
```julia
] add DPMMSubClusters
```

Or

```julia
julia> Pkg.add("DPMMSubClusters")
```

## Simple example to get you started

Start by setting the environment:

```julia
julia> using DPMMSubClusters
```

Continue by generating some random data, we will generate `10000` samples, taken from a mixture of 6 `2D` Gaussians, their mean sampled from a normal distribution with variance of 100:

```julia
julia> x,y,clusters = generate_gaussian_data(10000,2,6,100.0)
...
```

We will now run the model with a default `NIW` prior, burnout period of `10`, and with the GT we have generated:

```julia
julia> ret_values= fit(x,10.0, iters = 100,burnout = 10, gt = y)
Iteration: 1 || Clusters count: 1 || Log posterior: -70361.61688103084 || Vi score: 1.2054788914293095 || NMI score: 1.8419617838497155e-15 || Iter Time:0.005878925323486328 || Total time:0.005878925323486328
...
Iteration: 100 || Clusters count: 6 || Log posterior: -33552.84821500185 || Vi score: -0.0 || NMI score: 1.0 || Iter Time:0.013663053512573242 || Total time:1.0682177543640137
...
```

Now we can examine the labels by accessing the returned values:
```julia
julia> labels = ret_values[1]
4
4
4
4
4
4
4
⋮
2
2
2
2
2
2
```
