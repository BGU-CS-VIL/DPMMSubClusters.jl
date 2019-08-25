# DPMMSubClusters.jl

Package provides an easy, fast and scalable way to perform inference in Dirichlet Process Mixture Models.

[https://github.com/dinarior/DPMMSubClusters.jl](https://github.com/dinarior/DPMMSubClusters.jl)

Developed from the code of:

[ Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia](https://www.cs.bgu.ac.il/~dinari/papers/dpmm_hpml2019.pdf) by Dinari et al.

Which is based on the algorithm from:

[Parallel Sampling of DP Mixture Models using Sub-Clusters Splits](http://people.csail.mit.edu/jchang7/pubs/publications/chang13_NIPS.pdf) by Chang and Fisher.

The package currently supports Gaussian and Multinomial priors, however adding your own is very easy, and more will come in future releases.

Examples:

[2d Gaussian with plotting](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/2d_gaussian/gaussian_2d.ipynb)

[Image Segmentation](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/image_seg/dpgmm-superpixels.ipynb)

[Example of running from a params file, including saving and loading](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/save_load_model/save_load_example.ipynb)

If you use this package in your research, please cite the following:

```
@inproceedings{Dinari:CCGrid:2019,
  title={Distributed {MCMC} Inference in {Dirichlet} Process Mixture Models Using {Julia}},
  author={Dinari, Or and Angel, Yu and Freifeld, Oren and Fisher III, John W},
  booktitle={International Symposium on Cluster, Cloud and Grid Computing (CCGRID) Workshop on High Performance Machine Learning Workshop},
  year={2019}
}
```

For any questions: dinari@post.bgu.ac.il
Also available on Julia's Slack.

Contributions, feature requests, suggestion etc.. are welcomed.
