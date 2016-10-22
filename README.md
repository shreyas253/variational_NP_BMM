# variational_NP_BMM

INTRODUCTION
------------
The following is a toolbox that performs variational inferences of Baysina mixture models. The implimentation currently includes Gaussian mixture models and Von Mises-Fisher mixture models. The priors on the weight distribution include non-paramteric disributions such as the Dirichlet proecess and the Pitman-Yor process, and parametric distributions such as the Dirichlet distribution.

This also forms the source code related to our work 

1. Shreyas Seshadri, Ulpu Remes and Okko Rasanen: "Dirichlet process mixture models for clustering i-vector data", submitted.

2. Shreyas Seshadri, Ulpu Remes and Okko Rasanen: "Comparison of Non-parametric Bayesian Mixture Models for Zero-Resource Speech Processing", submitted.

Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi

Last updated: 22.10.2016


LICENSE
-------

Copyright (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasanen, Aalto University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The source code must be referenced when used in a published work.

METHODS
-------
The nonparametric priors included in the toolbox are constructed based on a stick-breaking process, and variational inference in the DPMM and PYPMM constructions is based on the truncated stick-breaking representation introduced in [1]. The variational posterior distribution over GMM parameters is calculated as proposed in [2] and the variational posterior distribution over VMFMM parameters as proposed in [3]. The variational method and numerical approximations applied in the VMFMM posterior estimation are also presented in the enclosed documentation [approximate variational inference in DPVMFMM](approximate-variational-inference.pdf).

FILES AND FUNCTIONS
-------------------
run.m: 
Script to run the toy data for various options of models and weight priors

VB_mixModel.m:
Main function that performs the variational inference of the Bayesian mixture models 

updateR.m:
Function to do the E-step in the variational inference algorithm

postUpdate.m:
Function to do the M-step in the variational inference algorithm

freeEnergyCalc.m:
Function to calculate the free energy of the variational distribution

approximate_bound.m:
Function to calculate the lower bound on the expected state likelihoods

d_besseli.m:
Function to calculate the approximate value of the Bessel function

wishartEntropy.m:
Function that calculates the entropy of the Wishart distribution

reorderFE.m:
Function to reorder the clusters in descending order of the cluster occupancy. Used for the non-parametric weight priors.

logdet.m:
Function to calculate the calculate the log determinant of x

logNormalize.m:
Function to normalize the values given in log scale

structMerge.m:
Function to merge the objects in 2 or 3 structures

plotClustering.m:
Function to plot the data and clustering of the 2D data

2d_data.mat and 10d_data.mat:
2 and 10 dimensional toy data


REFERENCES
----------
[1] D. M. Blei and M. I. Jordan. Variational inference for Dirichlet process mixtures, Bayesian analysis, vol. 1, no. 1, pp. 121-144, 2006.

[2] C. M. Bishop, Pattern recognition and machine learning. Springer, 2006.

[3] J. Taghia, Z. Ma, and A. Leijon. Bayesian estimation of the von-Mises Fisher mixture model with variational inference, IEEE TPAMI, vol. 36, no. 9, pp. 1701-1715, 2014.
