# hps_pytorch
A pytorch implementation of the HPS method for scattering problems. The intention is to make this code efficient for rapidly solving a large number of PDEs by taking advantage of hardware accelerators (GPUs).

## PDE Problem

Given a compactly-supported non-negative two-dimensional scattering potential $q(x)$,
the code solves this problem:

$$\Delta u(x) + \omega^2 (1 - q(x)) u(x) = 0 \ \ \ \ x \in \mathbb{R}^2$$

where the total wave field $u(x)$ is decomposed into the sum of a scattered wave field 
and an incident wave field $u(x) = u^{(i)}(x) + u^{(s)}(x)$. The incident wave field 
$u^{(i)}(x)$ is a plane wave with known direction and frequency $\omega$. 

## Reading 

These are the papers I've read about the HPS method:

 * (BGM15) [A spectrally accurate direct solution technique for frequency-domain scattering problems with variable media](https://link.springer.com/article/10.1007/s10543-014-0499-8)
 * (M15) [The Hierarchical Poincare-Steklov (HPS) solver for elliptic PDEs: A tutorial](https://arxiv.org/abs/1506.01308)

 Other references:

  * (T00) [Spectral Methods in MATLAB](https://epubs.siam.org/doi/book/10.1137/1.9780898719598)
  * (S11) [Numerical Analysis](https://press.princeton.edu/books/hardcover/9780691146867/numerical-analysis)


## TODO

 - [x] Build and test 1D Gauss-Legendre quadrature object.
 - [x] Re-write 2D Cheby quadrature object to include a list of indices mapping from 1d point locations to indices in the 2d point array. 
 - [x] Write tests for the 2D Cheby quadrature object
 - [x] Test the 1D differentiation matrix object.
 - [x] Write code to interpolate from equispaced grid to Chebyshev grid.
 - [x] Test code to interpolate from equispaced to Chebyshev grid.
 - [x] Build the `LeafNode` object.
 - [ ] Test the D_x and D_y operators by looking at a small example n=3.
 - [ ] Build objects for merging `LeafNode`s 

### TODO: Optimization
 - [ ] Precompute an interpolation matrix from a regularly-spaced grid to a Chebyshev grid. The code in `Cheby2D.interp_to_2d_points()` relies on scipy and does not take advantage of any precomputation.
 - [ ] Factor the stuff in `LeafNode.__init__()` to be all pre-computed and have the diagonals of certain objects be updated given a new scattering object or frequency.
 - [ ] Optimize the solve in `LeafNode.solve()`
 - [ ] There are multiple steps in `LeafNode` that require taking the Kronecker product between a dense matrix and 4x4 identity matrix. Can we accelerate applying these operators?



