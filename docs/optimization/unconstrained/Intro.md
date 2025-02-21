

# Introduction to Unconstrained Optimization

In unconstrained optimization, we aim to find a point  
```math
x^* \in \mathbb{R}^n
```
that minimizes a continuously differentiable function
```math
\min_{x \in \mathbb{R}^n} f(x).
```
That is, we seek $x^*$ such that
```math
f(x^*) \le f(x),\quad \forall\, x \in \mathbb{R}^n.
```
This framework underlies many algorithms in numerical optimization and forms the basis for further generalizations to constrained problems.

---

# 1. Problem Statement

We consider the general unconstrained optimization problem:
```math
\min_{x \in \mathbb{R}^n} f(x),
```
where $f : \mathbb{R}^n \to \mathbb{R}$ is assumed to be continuously differentiable (and, in many cases, twice continuously differentiable).

For example, a simple quadratic problem is
```math
f(x) = \frac{1}{2}x^T Q x - b^T x,
```
with $Q \in \mathbb{R}^{n \times n}$ symmetric and positive definite and $b \in \mathbb{R}^n$.

---

# 2. Taylor Expansion

To analyze and approximate the behavior of $f$ near a given point $x_k$, we use the Taylor expansion. For a twice continuously differentiable function, the second-order Taylor expansion about $x_k$ is
```math
f(x_k + p) \approx f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2 f(x_k) p,
```
where:
- $\nabla f(x_k)$ is the gradient at $x_k$,
- $\nabla^2 f(x_k)$ is the Hessian matrix at $x_k$,
- $p$ is a perturbation (or search) direction.

## 2.1. Why Use Taylor Expansion?

The Taylor expansion is used for several key reasons:
- **Local Approximation:** It provides a quadratic model of $f$ near $x_k$ that is easier to minimize than the full nonlinear function.
- **Derivation of Optimality Conditions:** The first-order term leads directly to the necessary condition for optimality: $\nabla f(x^*) = 0$.
- **Algorithm Design:** Many iterative methods (such as Newton’s method and quasi-Newton methods) use the quadratic model to compute search directions.

---

# 3. Optimality Conditions

Optimality conditions are necessary (and sometimes sufficient) for a point to be a minimizer.

## 3.1. First-Order Optimality Condition

For a differentiable function $f$, if $x^*$ is a local minimizer, then
```math
\nabla f(x^*) = 0.
```
This is a **first-order necessary condition**.

## 3.2. Second-Order Optimality Conditions

If $f$ is twice continuously differentiable, then in addition to $\nabla f(x^*) = 0$, a local minimizer must satisfy:
```math
\nabla^2 f(x^*) \succeq 0,
```
meaning that the Hessian is positive semidefinite at $x^*$. If $\nabla^2 f(x^*) \succ 0$ (positive definite), then $x^*$ is a strict local minimizer.

---

# 4. Local vs. Global Optima

- **Local Minimizer:** A point $x^*$ is a local minimizer if there exists a neighborhood $U$ such that
  ```math
  f(x^*) \le f(x), \quad \forall\, x \in U.
  ```
- **Global Minimizer:** A point $x^*$ is a global minimizer if
  ```math
  f(x^*) \le f(x), \quad \forall\, x \in \mathbb{R}^n.
  ```

For convex functions (see next section), every local minimizer is global. For nonconvex functions, however, multiple local minima may exist.

---

# 5. Convex and Concave Functions

- **Convex Function:**  
  A function $f : \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x, y \in \mathbb{R}^n$ and for all $\theta \in [0,1]$,
  ```math
  f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y).
  ```
  In convex optimization, any local minimizer is also a global minimizer.

- **Concave Function:**  
  A function is concave if $-f$ is convex. That is,
  ```math
  f(\theta x + (1-\theta)y) \ge \theta f(x) + (1-\theta)f(y).
  ```

Convexity plays a central role in the design and analysis of optimization algorithms. For instance, many algorithms (gradient descent, Newton’s method) have stronger convergence guarantees when applied to convex problems.

---

# 6. Examples in Julia and Python

Below are simple examples of solving a one-dimensional unconstrained optimization problem using steepest descent with a backtracking line search. The same example is shown in both Julia and Python.

### 6.1. Example Problem

Consider the function
```math
f(x) = (x-5)^2,
```
which is convex with a unique global minimizer at $x^*=5$.

---

### 6.2. Example

=== "julia"
```julia 
# Define the objective function and its gradient
f(x) = (x - 5)^2
grad_f(x) = 2*(x - 5)

# Backtracking line search function
function backtracking_line_search(f, grad_f, x, p; α_init=1.0, τ=0.5, c=1e-4)
    α = α_init
    f_x = f(x)
    g_x = grad_f(x)
    # Ensure p is a descent direction
    if dot(g_x, p) ≥ 0
        error("Search direction is not a descent direction.")
    end
    while f(x + α*p) > f_x + c*α*dot(g_x, p)
        α *= τ
    end
    return α
end

# Steepest descent iteration (single step)
x0 = 0.0
p = -grad_f(x0)  # steepest descent direction
α = backtracking_line_search(f, grad_f, x0, p)
x1 = x0 + α*p

println("Julia: x0 = ", x0)
println("Julia: Step size α = ", α)
println("Julia: New iterate x1 = ", x1)
```

=== "python"
```python
import numpy as np

def f(x):
    return (x - 5)**2

def grad_f(x):
    return 2*(x - 5)

def backtracking_line_search(f, grad_f, x, p, α_init=1.0, τ=0.5, c=1e-4):
    α = α_init
    f_x = f(x)
    g_x = grad_f(x)
    # Ensure p is a descent direction
    if np.dot(g_x, p) >= 0:
        raise ValueError("Search direction is not a descent direction.")
    while f(x + α * p) > f_x + c * α * np.dot(g_x, p):
        α *= τ
    return α

# Steepest descent iteration (single step)
x0 = 0.0
p = -grad_f(x0)  # steepest descent direction
α = backtracking_line_search(f, grad_f, x0, p)
x1 = x0 + α * p

print("Python: x0 =", x0)
print("Python: Step size α =", α)
print("Python: New iterate x1 =", x1)
```

# 7. Additional Topics and Further Reading

For further study, you are encouraged to explore:

- **MINRES and Other Krylov Methods:**  
  These methods are used for solving large-scale linear systems and appear in second-order methods such as Newton–MR.  
  [Learn more about MINRES](#) *(link to a dedicated MINRES lecture note)*

- **Line Search Techniques:**  
  Inexact line search methods (including the Wolfe and strong Wolfe conditions) are central to ensuring robust convergence in many algorithms.  
  [Learn more about Line Search](#) *(link to a dedicated line search lecture note)*

- **Global vs. Local Optimality:**  
  Understanding when a local minimum is also global (e.g. in convex problems) is crucial for algorithm design and analysis.

- **Convexity and Concavity:**  
  These properties not only affect the quality of solutions but also the convergence guarantees of optimization algorithms.

- **Taylor Expansion and Its Role in Optimization:**  
  The quadratic approximation provided by Taylor’s expansion underpins many optimization methods.

---

# References

1. **Nocedal, Jorge, and Stephen J. Wright.**  
   *Numerical Optimization.* Springer, 2006.
2. **Armijo, Larry.**  
   "Minimization of Functions Having Lipschitz Continuous First Partial Derivatives."  
   *Pacific Journal of Mathematics*, 1966.
3. **Wolfe Conditions – Wikipedia.**  
   [https://en.wikipedia.org/wiki/Wolfe_conditions](https://en.wikipedia.org/wiki/Wolfe_conditions)
4. **Backtracking Line Search – Wikipedia.**  
   [https://en.wikipedia.org/wiki/Backtracking_line_search](https://en.wikipedia.org/wiki/Backtracking_line_search)
5. **LineSearches.jl Documentation.**  
   [https://github.com/JuliaNLSolvers/LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl)
6. **JSOSolvers.jl.**  
   [https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl)

