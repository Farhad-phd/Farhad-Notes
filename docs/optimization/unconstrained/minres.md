

# Comprehensive Lecture on MINRES and Newton-MR
 It explores advanced Krylov subspace methods for symmetric systems—specifically MINRES, CG, and CR—and shows how these methods are incorporated into a Newton-type method (Newton-MR) for nonconvex smooth unconstrained optimization. Special emphasis is placed on detecting negative curvature via MINRES and the strategy to switch to a line search when such curvature is encountered.

---

## 1. Overview of Krylov Subspace Methods for Symmetric Systems

When solving a linear system

```math
Ax = b,
```

iterative methods based on Krylov subspaces are preferred for large-scale or sparse problems. In symmetric settings, three key methods are:

- **Conjugate Gradient (CG):**  
  Designed for symmetric positive definite (SPD) systems. CG minimizes the energy (or \(A\)-norm) error in the Krylov subspace. Its convergence rate is \(O(\sqrt{\kappa(A)})\) in the best case, where \(\kappa(A)\) is the condition number.

- **Conjugate Residual (CR):**  
  Applicable to symmetric systems—including indefinite ones—CR minimizes the residual norm \(\|b-Ax\|\) over the Krylov subspace. While it is closely related to MINRES, its recurrences require slightly more storage.

- **MINRES (Minimum Residual Method):**  
  Tailored for symmetric systems (even if indefinite), MINRES directly minimizes the 2-norm of the residual
  ```math
  r(x) = b - Ax,
  ```
  by projecting the problem onto a Krylov subspace generated via the Lanczos process—a specialized Arnoldi iteration for symmetric matrices. MINRES is especially attractive because it efficiently handles negative or zero curvature situations that arise in nonconvex optimization.

> **Key Remark:**  
> While CG is optimal for SPD systems, MINRES extends to indefinite matrices and inherently detects directions of non-positive curvature.

*For additional background on MINRES, see [Paige and Saunders (1975)](https://doi.org/10.1137/0712047) and the Wikipedia article on [Minimal Residual Method](https://en.wikipedia.org/wiki/Minimal_residual_method).*

---

## 2. In-Depth: The MINRES Method

### 2.1. Derivation via the Lanczos Process

For a symmetric matrix \(A\), the Arnoldi process simplifies to the Lanczos iteration. Starting with the normalized residual

```math
q_1 = \frac{r_0}{\|r_0\|},
```

the Lanczos process generates an orthonormal basis

```math
Q_m = [q_1, q_2, \dots, q_m]
```

for the Krylov subspace

```math
\mathcal{K}_m(A, r_0) = \operatorname{span}\{r_0, Ar_0, A^2r_0, \dots, A^{m-1}r_0\}.
```

In exact arithmetic, one obtains the relation

```math
A Q_m = Q_{m+1} \tilde{T}_m,
```

where \(\tilde{T}_m\) is an \((m+1) \times m\) tridiagonal matrix. Any approximate solution in the affine subspace \(x_0 + \mathcal{K}_m\) can then be written as

```math
x_m = x_0 + Q_m y,
```

with \(y \in \mathbb{R}^m\). The residual becomes

```math
r_m = b - Ax_m = r_0 - A Q_m y = \|r_0\| \left(e_1 - \tilde{T}_m y\right),
```

so that minimizing \(\|r_m\|\) is equivalent to solving a small least-squares problem:

```math
y_m = \operatorname*{argmin}_{y\in\mathbb{R}^m} \left\|\|r_0\|e_1 - \tilde{T}_m y\right\|_2.
```

### 2.2. Convergence Properties

For symmetric (even indefinite) matrices, convergence bounds for MINRES can be derived via polynomial approximation. For example, if the eigenvalues of \(A\) are divided into positive and negative parts, one may have a bound of the form

```math
\frac{\|r_m\|}{\|r_0\|} \le \left(\frac{\sqrt{\kappa_+ \kappa_-} - 1}{\sqrt{\kappa_+ \kappa_-} + 1}\right)^{\lfloor m/2 \rfloor},
```

where
- \(\kappa_+ = \frac{\max_{\lambda \in \Lambda_+}|\lambda|}{\min_{\lambda \in \Lambda_+}|\lambda|}\) (for positive eigenvalues), and  
- \(\kappa_- = \frac{\max_{\lambda \in \Lambda_-}|\lambda|}{\min_{\lambda \in \Lambda_-}|\lambda|}\) (for negative eigenvalues).

For SPD matrices, one recovers a similar bound to that for CG:

```math
\frac{\|r_m\|}{\|r_0\|} \le 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^m.
```

These bounds illustrate that the convergence rate depends on the spectral properties of \(A\); clustering of eigenvalues near zero can slow convergence.

---

## 3. Comparison: MINRES vs. CG vs. CR

| Method   | Matrix Requirements             | Minimization Target                          | Recurrence Complexity         | Key Strengths                                              |
|----------|---------------------------------|----------------------------------------------|-------------------------------|------------------------------------------------------------|
| **CG**   | Symmetric positive definite     | Energy norm error \(\|x - x^*\|_A\)           | Short recurrence (\(O(1)\) per iteration)  | Highly efficient for SPD systems; optimal for quadratic minimization. |
| **CR**   | Symmetric (SPD or indefinite)   | 2-norm residual \(\|b - Ax\|\)                | Similar to CG with slightly more storage  | Minimizes the residual; closely related to MINRES.         |
| **MINRES** | Symmetric (including indefinite) | Direct minimization of \(\|b - Ax\|_2\)       | Short recurrence via Lanczos (comparable to CG) | Robust for indefinite systems; inherently detects negative curvature.  |

In many optimization applications—especially within a Newton-type method—the Hessian \(H_k\) may be indefinite. In such cases, CG may not be applicable or requires modifications, whereas MINRES naturally handles these situations.

---

## 4. Newton-MR: A Newton-Type Method for Nonconvex Optimization

Newton-MR is a variant of Newton’s method that employs MINRES to solve the Newton system in nonconvex optimization problems. This is particularly useful when the Hessian

```math
H_k = \nabla^2 f(x_k)
```

is indefinite.

### 4.1. Motivation

- **Indefinite Hessians in Nonconvex Problems:**  
  In nonconvex settings, the Hessian can have negative or zero eigenvalues. Traditional Newton methods (or Newton-CG) may fail or require modifications such as adding a damping term, which may slow down convergence.

- **Leveraging MINRES:**  
  MINRES is designed for symmetric indefinite systems. It naturally minimizes the residual norm and, via its Lanczos process, can detect directions of negative curvature. This detection is key to ensuring that the algorithm does not take a full Newton step that could lead to ascent or non-descent.

### 4.2. Detecting Negative or Zero Curvature in MINRES

During the MINRES iterations, the Lanczos process computes recurrence coefficients \(\alpha_j\) (diagonal entries) and \(\beta_j\) (off-diagonal entries) that form the tridiagonal matrix \(\tilde{T}_m\). These coefficients carry curvature information:
  
- **Rayleigh Quotient Approximation:**  
  The diagonal coefficients \(\alpha_j\) approximate the Rayleigh quotient in the Krylov subspace. If at any iteration \(j\) an \(\alpha_j\) becomes non-positive (or nearly zero), it signals that the Hessian has a direction of non-positive (negative or zero) curvature.

- **Practical Detection:**  
  In implementations, MINRES monitors these recurrence coefficients. Once a coefficient satisfies
  ```math
  \alpha_j \leq \delta \quad (\delta \approx 0),
  ```
  this serves as a flag that a negative curvature direction has been encountered.

### 4.3. Switching to a Line Search

When negative or zero curvature is detected, Newton-MR adopts a safeguard by switching to a line search along the direction associated with the detected curvature:

1. **Extract Negative Curvature Direction:**  
   When the MINRES iteration indicates non-positive curvature, the algorithm identifies a direction \(d\) (often related to the most recent Krylov basis vector) that exhibits negative curvature.

2. **Line Search Procedure:**  
   Instead of using the full Newton step \(p\) computed via MINRES, the algorithm performs a line search along \(d\) to ensure that the update
   ```math
   x_{k+1} = x_k + \alpha d
   ```
   yields a sufficient decrease in the objective function \(f\).

3. **Ensuring Sufficient Decrease:**  
   A common strategy is to use an Armijo condition:
   ```math
   f(x_k + \alpha d) \le f(x_k) + c \, \alpha \, g_k^T d,
   ```
   with \(c \in (0,1)\). This safeguards the descent property even when the Hessian is indefinite.

4. **Adaptive Strategy:**  
   The paper details an adaptive strategy whereby the MINRES subproblem is solved only until a curvature condition is violated. At that point, the algorithm “switches off” the standard Newton update and reverts to a safeguarded line search, ensuring overall descent and robust convergence.

---

## 5. Detailed Pseudocode for Newton-MR with Curvature Detection

```markdown
Algorithm Newton-MR:
Input: Initial point \(x_0\), tolerance \(\epsilon\), maximum iterations \(K\).

for \(k = 0, 1, \dots, K-1\):
  1. Compute gradient: \(g_k = \nabla f(x_k)\).
  2. If \(\|g_k\| \le \epsilon\), terminate.
  3. Compute Hessian: \(H_k = \nabla^2 f(x_k)\).
  4. Solve \(H_k p = -g_k\) approximately using MINRES:
     a. Start with \(p^{(0)} = 0\).
     b. Run MINRES, and at each iteration monitor the Lanczos coefficient \(\alpha_j\).
     c. If any \(\alpha_j \le \delta\) (indicating negative/zero curvature), extract the corresponding direction \(d\).
  5. **If negative curvature detected:**
     - Switch to a safeguarded line search along \(d\) (instead of the full Newton step).
  6. **Else:**
     - Accept the computed \(p\) as the Newton direction.
  7. Perform a line search to determine step size \(\alpha_k\) that satisfies a sufficient decrease condition.
  8. Update \(x_{k+1} = x_k + \alpha_k \times\) (chosen direction).
end for
```

---

## 6. Illustrative Julia Implementation

Below is a Julia code snippet that outlines the Newton-MR algorithm. This version includes comments on detecting negative curvature and switching to a line search. (In practice, more advanced curvature-detection logic may be integrated.)

```julia
using LinearAlgebra, IterativeSolvers

"""
    newtonMR(f, grad, hess, x0; tol, maxit)

Newton-MR solver for unconstrained optimization.
- `f`: objective function
- `grad`: gradient function
- `hess`: Hessian function
- `x0`: initial point
- `tol`: tolerance on the gradient norm
- `maxit`: maximum number of outer iterations
"""
function newtonMR(f, grad, hess, x0; tol=1e-6, maxit=50)
    x = x0
    for k in 1:maxit
        g = grad(x)
        println("Iteration $k: ||grad|| = ", norm(g))
        if norm(g) < tol
            println("Convergence achieved at iteration $k")
            return x
        end
        
        H = hess(x)
        # Define Hessian operator for MINRES
        H_operator(v) = H * v
        
        # Solve H * p = -g using MINRES; monitor MINRES coefficients.
        # (In a full implementation, one would check the recurrence coefficients here.)
        p, flag = minres(H_operator, -g, tol=1e-4, maxiter=length(x))
        if flag != 0
            println("MINRES did not converge at iteration $k; flag = $flag")
            break
        end
        
        # Here we assume that our MINRES routine internally monitors the Lanczos recurrence.
        # If negative curvature is detected (e.g., a coefficient falls below a threshold δ),
        # the algorithm would set a flag and extract the corresponding direction d.
        # For this illustrative code, we simply check a condition (dummy check):
        negative_curvature_detected = false  # Replace with actual detection logic
        if negative_curvature_detected
            println("Negative curvature detected at iteration $k. Switching to line search.")
            # Assume d is the extracted negative curvature direction (here, simply use p)
            d = p  # In practice, d would be chosen based on the curvature information
        else
            d = p
        end
        
        # Backtracking line search using Armijo condition:
        α = 1.0
        while f(x + α * d) > f(x) - 1e-4 * α * dot(g, d)
            α *= 0.5
        end
        
        x += α * d
    end
    return x
end

# Example: quadratic function (SPD case; for illustration)
f(x) = 0.5 * dot(x, x)         # f(x) = 0.5 xᵀx
grad(x) = x                    # grad f(x) = x
hess(x) = I                    # Hessian is the identity

x0 = randn(10)
solution = newtonMR(f, grad, hess, x0)
println("Solution: ", solution)
```

*Note:* In real nonconvex applications, the Hessian may be indefinite. The MINRES routine would then monitor the Lanczos coefficients (\(\alpha_j\)) to detect negative curvature (when any \(\alpha_j\) falls below a small threshold \(\delta\)). Upon such detection, the algorithm switches from the full Newton step to a safeguarded line search along the direction associated with the negative curvature, ensuring a sufficient decrease in \(f\).

---

## 7. Concluding Remarks

- **MINRES as a Robust Subproblem Solver:**  
  MINRES directly minimizes the residual norm and is well suited for symmetric indefinite matrices. Its derivation via the Lanczos process yields efficient short recurrences and, crucially, the ability to detect directions of non-positive curvature.

- **Comparative Insights:**  
  While CG is optimal for SPD systems, MINRES (and CR) extend applicability to broader classes of symmetric matrices. Their convergence properties—governed by the distribution of eigenvalues—illustrate how negative curvature can be detected and exploited.

- **Newton-MR Advantages:**  
  Newton-MR leverages MINRES to solve the Newton system even when the Hessian is indefinite. When MINRES detects negative or zero curvature (via non-positive Lanczos coefficients), the algorithm switches to a line search along a negative curvature direction, ensuring descent and robust convergence. This adaptive mechanism underpins the complexity guarantees provided by the method.

---

## References

1. **Liu, Yang, and Fred Roosta.**  
   *A Newton-MR Algorithm with Complexity Guarantees for Nonconvex Smooth Unconstrained Optimization.*  
   ArXiv preprint, [arXiv:2208.07095](https://arxiv.org/abs/2208.07095). citeturn0search1

2. **Paige, C. C., and M. A. Saunders.**  
   *Solution of Sparse Indefinite Systems of Linear Equations.*  
   SIAM Journal on Numerical Analysis, 12 (1975): 617–629.

3. **Saad, Y.**  
   *Iterative Methods for Sparse Linear Systems.*  
   SIAM, 2003.

4. **Wikipedia.**  
   *Minimal Residual Method.*  
   [https://en.wikipedia.org/wiki/Minimal_residual_method](https://en.wikipedia.org/wiki/Minimal_residual_method) (Accessed: November 2023). citeturn0search0

5. **Fong, D. C.-L., and M. A. Saunders.**  
   *MINRES: A Krylov Subspace Method for Symmetric Linear Systems.*  
   (For further background on MINRES.)

