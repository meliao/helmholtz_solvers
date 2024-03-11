# helmholtz_solvers
Pytorch implementations of high-resolution solvers for forward scattering problems. The intention is to make this code efficient for rapidly solving a large number of PDEs by taking advantage of hardware accelerators (GPUs).

## PDE Problem

Given a compactly-supported non-negative two-dimensional scattering potential $q(x)$,
the code solves this problem:

$$\Delta u(x) + \omega^2 (1 - q(x)) u(x) = 0 \ \ \ \ x \in \mathbb{R}^2$$

where the solution $u(x) = u_s(x) + u_i(x)$. The incident part of the wave field is a plane wave with known direction and frequency $\omega$. The scattered part of the wave field satisfies the Sommerfeld radiation condition: 

$$ \frac{\partial u_s(x)}{\partial \| x \| } - i \omega u_s(x) = o\left( \| x \|^{-1/2} \right) \ \ \ \| x \| \to \infty $$


## Solver 1: Lippmann-Schwinger Equation (Stable)

Discretizes and solves the following integral equation:

$$ \sigma(x) - \omega^2(x) q(x) \int G_\omega(x, x') \sigma(x') dx' = \omega^2 u_i(x) q(x)$$

where $G_\omega(x,x')$ is the Green's function for the homogeneous Helmholtz operator $\Delta + \omega^2$. The solution $u_s(x)$ to the above problem is given by $u_s(x) = \int G_\omega(x, x') \sigma(x') dx'$.

The linear equations are discretized and implemented as sparse linear operators, and the system is inverted using an iterative solver GMRES or (optionally) BICGSTAB. 

The object `HelmholtzSolverAccelerated` in the file `src/lippmann_schwinger_eqn/HelmholtzSolver.py` serves as the main interface. 

The object should be initialized using the `setup_accelerated_solver()` method. Some of the methods that may be useful: 
 * `HemholtzSolverAccelerated.Helmholtz_solve_exterior()`: Given incident source directions and a scattering object, evalutates the scattered wave field $u_s(x)$ on a ring far away from the scattering object. This is often used as a forward model in inverse wave scattering problems.
 * `HelmholtzSolverAccelerated.Helmholtz_solve_interior()`: Given incident source directions and a scattering object, evaluates the solution $u(x) = u_s(x) + u_i(x)$ on the scattering domain. Returns the total wave field, the incident wave field, and the scattered wave field.


## Solver 2: Hierarchical Poincare-Steklov (Under Development)

The HPS method uses a heirarchical spatial decomposition to define and directly solve a linear system defined on the boundary of the scattering domain.


### Reading 

These are the papers I've read about the HPS method:

 * (BGM15) [A spectrally accurate direct solution technique for frequency-domain scattering problems with variable media](https://link.springer.com/article/10.1007/s10543-014-0499-8)
 * (M15) [The Hierarchical Poincare-Steklov (HPS) solver for elliptic PDEs: A tutorial](https://arxiv.org/abs/1506.01308)

 Other references:

  * (T00) [Spectral Methods in MATLAB](https://epubs.siam.org/doi/book/10.1137/1.9780898719598)
  * (S11) [Numerical Analysis](https://press.princeton.edu/books/hardcover/9780691146867/numerical-analysis)


### TODO

 - [x] Build and test 1D Gauss-Legendre quadrature object.
 - [x] Re-write 2D Cheby quadrature object to include a list of indices mapping from 1d point locations to indices in the 2d point array. 
 - [x] Write tests for the 2D Cheby quadrature object
 - [x] Test the 1D differentiation matrix object.
 - [x] Write code to interpolate from equispaced grid to Chebyshev grid.
 - [x] Test code to interpolate from equispaced to Chebyshev grid.
 - [x] Build the `LeafNode` object.
 - [x] Test the D_x and D_y operators by differentiating polynomials.
 - [ ] Test the `LeafNode` object by building a notebook and confirming that it produces acceptable local solutions to the scattering problem.
 - [ ] Build objects for merging `LeafNode`s 
 - [ ] Refactor `LeafNode` to have a parent class `Node`. Make `Merge` operate on two `Node`s.

#### TODO: Optimization
 - [ ] Precompute an interpolation matrix from a regularly-spaced grid to a Chebyshev grid. The code in `Cheby2D.interp_to_2d_points()` relies on scipy and does not take advantage of any precomputation.
 - [ ] Factor the stuff in `LeafNode.__init__()` to be all pre-computed and have the diagonals of certain objects be updated given a new scattering object or frequency.
 - [ ] Optimize the lineaer system solve in `LeafNode.solve()`
 - [ ] There are multiple steps in `LeafNode` that require taking the Kronecker product between a dense matrix and 4x4 identity matrix. Can we accelerate applying these operators?



