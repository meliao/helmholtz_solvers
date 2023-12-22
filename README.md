# hps_pytorch
A pytorch implementation of the HPS method for scattering problems. The intention is to make this code efficient for rapidly solving a large number of PDEs by taking advantage of hardware accelerators (GPUs).

## PDE Problem

Given a compactly-supported non-negative two-dimensional scattering potential $q(x)$,
the code solves this problem:

$$\Delta u(x) + \omega^2 (1 - q(x)) u(x) = 0 \ \ \ \ x \in \R^2$$

where the total wave field $u(x)$ is decomposed into the sum of a scattered wave field 
and an incident wave field $u(x) = u^{(i)}(x) + u^{(s)}(x)$. The incident wave field 
$u^{(i)}(x)$ is a plane wave with known direction and frequency $\omega$. 

## Reading 

These are the papers I've read about the HPS method:

 * (BGM15) [A spectrally accurate direct solution technique for frequency-domain scattering problems with variable media](https://link.springer.com/article/10.1007/s10543-014-0499-8)
 * (M15) [The Hierarchical Poincare-Steklov (HPS) solver for elliptic PDEs: A tutorial](https://arxiv.org/abs/1506.01308)


## TODO

 - [x] Build and test 1D Gauss-Legendre quadrature object.
 - [x] Re-write 2D Cheby quadrature object to include a list of indices mapping from 1d point locations to indices in the 2d point array. 
 - [x] Write tests for the 2D Cheby quadrature object
 - [ ] Test the 1D differentiation matrix object.
 - [ ] Build and test LeafNode object.


