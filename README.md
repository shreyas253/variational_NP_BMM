# variational_NP_BMM

INTRODUCTION
------------
The following is a toolbox that performs variational inferences of Baysina mixture models. The implimentation currently includes Gaussian mixture models and Von Mises-Fisher mixture models. The priors on the weight distribution include non-paramteric disributions such as the Dirichlet proecess and the Pitman-Yor process, and parametric distributions such as the Dirichlet distribution.

The also forms the source code related to our work 

1. Shreyas Seshadri, Ulpu Remes and Okko Rasanen: "Dirichlet process mixture models for clustering i-vector data", submitted.

2. Shreyas Seshadri, Ulpu Remes and Okko Rasanen: "Comparison of Non-parametric Bayesian Mixture Models for Zero-Resource Speech Processing", submitted.

Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi

Last updated: 19.10.2016


LICENSE
-------

Copyright (C) 2014 Shreyas Seshadri, Ulpu Remes and Okko Rasanen, Aalto University

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
detailed explanation coming soon!!

REFERENCES
----------
[1] D. M. Blei and M. I. Jordan. Variational inference for Dirichlet process mixtures, Bayesian analysis, vol. 1, no. 1, pp. 121-144, 2006.

[2] C. M. Bishop, Pattern recognition and machine learning. Springer, 2006.

[3] J. Taghia, Z. Ma, and A. Leijon. Bayesian estimation of the von-Mises Fisher mixture model with variational inference, IEEE TPAMI, vol. 36, no. 9, pp. 1701-1715, 2014.
