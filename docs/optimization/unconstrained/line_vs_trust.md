

# Two Strategies in Unconstrained Optimization: Line Search vs. Trust Region

In iterative optimization, after determining a search direction $ p_k $ from the current iterate $ x_k $, one must decide how far to move along that direction. Two broad strategies to determine the step are:

1. **Line Search Methods:**  
   Find a step size $\alpha_k$ such that the new iterate is
   ```math
   x_{k+1} = x_k + \alpha_k p_k,
   ```
   and $\alpha_k$ is chosen to guarantee sufficient decrease (or satisfy other conditions).

2. **Trust Region Methods:**  
   Instead of moving along a fixed direction with a step size, these methods first build a local model $ m_k(p) $ of the objective function around $ x_k $ and then solve a subproblem that restricts the step $ p $ to lie within a neighborhood (trust region) of $ x_k $. The update is then
   ```math
   x_{k+1} = x_k + p_k, \quad \text{with } p_k = \arg\min_{\|p\|\le \Delta_k} m_k(p),
   ```
   where $\Delta_k$ is the trust-region radius.

Below we explore each strategy in more detail.

---

## 1. Line Search Methods

### 1.1. Concept and Motivation

In a line search method, given a descent direction $ p_k $, we define the one-dimensional function
```math
\phi(\alpha) = f(x_k + \alpha p_k),
```
and seek a step length $\alpha_k$ that (approximately) minimizes $\phi$. However, rather than computing the exact minimizer, **inexact line search** methods impose conditions to guarantee sufficient progress. A common requirement is the **Armijo condition**:
```math
f(x_k + \alpha_k p_k) \le f(x_k) + c \alpha_k \nabla f(x_k)^T p_k,
```
with a small constant $c \in (0,1)$ (often $c \approx 10^{-4}$).

To further prevent $\alpha_k$ from being too small, **Wolfe conditions** (or strong Wolfe conditions) add a curvature condition:
```math
\big| \nabla f(x_k + \alpha_k p_k)^T p_k \big| \le c_2 \big| \nabla f(x_k)^T p_k \big|,
```
with $c_2$ typically in $(c, 1)$.

### 1.2. Algorithm: Backtracking Line Search

A common practical approach is the backtracking line search:
1. **Initialize:** Set $\alpha = \alpha_{\text{init}}$ (often 1) and choose $ \tau \in (0,1) $ (e.g., 0.5).
2. **Iterate:** While the Armijo condition is not met, update $\alpha \leftarrow \tau \alpha$.
3. **Return:** Use the final $\alpha$ as $\alpha_k$.

### 1.3. Code Examples


=== julia
```julia
# Define the objective function and its gradient.
f(x) = (x - 5)^2
grad_f(x) = 2*(x - 5)

# Backtracking line search function.
function backtracking_line_search(f, grad_f, x, p; α_init=1.0, τ=0.5, c=1e-4)
    α = α_init
    f_x = f(x)
    g_x = grad_f(x)
    # Ensure p is a descent direction.
    if dot(g_x, p) ≥ 0
        error("Search direction is not a descent direction.")
    end
    while f(x + α*p) > f_x + c*α*dot(g_x, p)
        α *= τ
    end
    return α
end

# Example: Steepest descent for f(x) = (x-5)^2.
x0 = 0.0
p = -grad_f(x0)  # Steepest descent direction.
α = backtracking_line_search(f, grad_f, x0, p)
x1 = x0 + α * p

println("Julia: x0 = ", x0)
println("Julia: Step size α = ", α)
println("Julia: New iterate x1 = ", x1)
```

<!-- #### Python Example -->

=== python

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
    # Ensure p is a descent direction.
    if np.dot(g_x, p) >= 0:
        raise ValueError("Search direction is not a descent direction.")
    while f(x + α * p) > f_x + c * α * np.dot(g_x, p):
        α *= τ
    return α

# Example: Steepest descent for f(x) = (x-5)^2.
x0 = 0.0
p = -grad_f(x0)  # Steepest descent direction.
α = backtracking_line_search(f, grad_f, x0, p)
x1 = x0 + α * p

print("Python: x0 =", x0)
print("Python: Step size α =", α)
print("Python: New iterate x1 =", x1)
```

---

## 2. Trust Region Methods

### 2.1. Concept and Motivation

Trust region methods approach the step selection problem differently. Instead of choosing a step size along a fixed direction, these methods:
- **Build a local model:** Typically a quadratic model of $ f $ around $ x_k $:
  ```math
  m_k(p) = f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T B_k p,
  ```
  where $ B_k $ is either the Hessian $\nabla^2 f(x_k)$ or an approximation thereof.
- **Restrict the step:** Only consider steps $ p $ that lie within a neighborhood of $ x_k $ of radius $\Delta_k$:
  ```math
  \|p\| \le \Delta_k.
  ```
- **Solve the subproblem:** Find
  ```math
  p_k = \arg\min_{\|p\| \le \Delta_k} m_k(p).
  ```
- **Update the iterate:** Set
  ```math
  x_{k+1} = x_k + p_k.
  ```

The idea is that the quadratic model is assumed to be a good approximation within the "trust region." If the model predicts a good decrease and the actual function decreases sufficiently, the trust region can be expanded; otherwise, it is contracted.

### 2.2. Key Components

- **Trust Region Subproblem:**  
  The quadratic subproblem is
  ```math
  \min_{p} \; m_k(p) \quad \text{subject to} \quad \|p\| \le \Delta_k.
  ```
  This subproblem is usually solved approximately (e.g., via the dogleg method, conjugate gradient, or truncated CG).

- **Trust Region Radius $\Delta_k$:**  
  After computing $ p_k $, we compare the actual reduction in $ f $ to the predicted reduction by $ m_k(p) $. Based on the ratio
  ```math
  \rho_k = \frac{f(x_k) - f(x_k+p_k)}{m_k(0) - m_k(p_k)},
  ```
  we adjust $\Delta_k$:
  - If $\rho_k$ is high (model was good), increase $\Delta_k$.
  - If $\rho_k$ is low (model was poor), decrease $\Delta_k$.

### 2.3. Code Sketch

Below is a simplified pseudocode outline and code snippet in Julia for a trust region step.

**Pseudocode:**
1. At iterate $x_k$, build a quadratic model:
   ```math
   m_k(p) = f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T B_k p.
   ```
2. Solve the subproblem:
   ```math
   p_k = \arg\min_{\|p\| \le \Delta_k} m_k(p).
   ```
3. Compute the ratio:
   ```math
   \rho_k = \frac{f(x_k) - f(x_k+p_k)}{m_k(0) - m_k(p_k)}.
   ```
4. Update $x_{k+1} = x_k + p_k$ and adjust $\Delta_k$ based on $\rho_k$.

#### Example (Pseudocode Style)

=== julia
```julia
function trust_region_step(f, grad_f, B, x, Δ)
    # Define the quadratic model:
    m(p) = f(x) + dot(grad_f(x), p) + 0.5 * dot(p, B * p)
    
    # (For illustration, assume we solve the subproblem exactly; in practice, use a solver.)
    # Here we use the dogleg method or a truncated CG method.
    p = solve_trust_region_subproblem(grad_f(x), B, Δ)
    
    # Compute actual and predicted reduction:
    actual_reduction = f(x) - f(x + p)
    predicted_reduction = m(zeros(length(x))) - m(p)
    ρ = actual_reduction / predicted_reduction
    
    # Update trust region radius (simplified rule):
    if ρ < 0.25
        Δ_new = 0.5 * Δ
    elseif ρ > 0.75 && norm(p) == Δ
        Δ_new = min(2 * Δ, Δ_max)  # assume some Δ_max is defined
    else
        Δ_new = Δ
    end
    return x + p, Δ_new, ρ
end
```

*Note:* In an actual implementation, the subproblem solver (e.g. dogleg or truncated CG) would be used to compute $p$. Packages like [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) and [Optimization.jl](https://docs.sciml.ai/Optimization/) provide robust trust region methods.

<!-- #### Python Example (Simplified) -->

=== python
```python
import numpy as np

def quadratic_model(f, grad, B, x, p):
    return f(x) + np.dot(grad(x), p) + 0.5 * np.dot(p, B @ p)

def trust_region_step(f, grad, B, x, Δ, Δ_max=1.0):
    # For illustration, assume we solve the trust region subproblem approximately.
    # Here we use the Cauchy point as a simple solution:
    g = grad(x)
    p_cauchy = - (Δ / np.linalg.norm(g)) * g
    p = p_cauchy  # In practice, use dogleg or truncated CG.
    
    actual_reduction = f(x) - f(x + p)
    predicted_reduction = quadratic_model(f, grad, B, x, np.zeros_like(x)) - quadratic_model(f, grad, B, x, p)
    ρ = actual_reduction / predicted_reduction
    
    if ρ < 0.25:
        Δ_new = 0.5 * Δ
    elif ρ > 0.75 and np.linalg.norm(p) >= Δ:
        Δ_new = min(2 * Δ, Δ_max)
    else:
        Δ_new = Δ
    return x + p, Δ_new, ρ

# Example: minimize f(x) = (x-5)^2, with constant Hessian B = 2.
f = lambda x: (x - 5)**2
grad = lambda x: 2*(x - 5)
B = np.array([[2.0]])

x0 = np.array([0.0])
Δ = 1.0
x1, Δ_new, ρ = trust_region_step(f, grad, B, x0, Δ)
print("Python: New iterate x1 =", x1)
print("Python: New trust region radius Δ =", Δ_new)
print("Python: Ratio ρ =", ρ)
```

---

## 3. Comparison: Line Search vs. Trust Region

- **Line Search:**
  - **Idea:** Move along a fixed search direction $ p_k $ by choosing a step size $\alpha_k$.
  - **Key Conditions:** Armijo and Wolfe (or strong Wolfe) conditions.
  - **Pros:** Often simple to implement; works well when the local model is valid along the entire line.
  - **Cons:** Requires careful selection of $\alpha_{\text{init}}$ and may involve multiple function evaluations per iteration.

- **Trust Region:**
  - **Idea:** Build a local (often quadratic) model $ m_k(p) $ and trust that model only within a ball (trust region) of radius $\Delta_k$. The step $ p_k $ is chosen by approximately minimizing $ m_k(p) $ subject to $\|p\|\leq \Delta_k$.
  - **Pros:** Naturally limits the step size when the model is unreliable; robust in the presence of nonconvexity.
  - **Cons:** The subproblem may be harder to solve; requires mechanisms to adjust the radius $\Delta_k$.

Both strategies are widely used. The choice depends on the problem structure and practical considerations. In many modern solvers, hybrid strategies exist, and libraries like JSOSolvers.jl offer both line search and trust region solvers for unconstrained optimization.

---

## 4. Further Reading and References

For a deeper dive into these strategies and their analysis, consult:

- **Nocedal, Jorge, and Stephen J. Wright.**  
  *Numerical Optimization.* Springer, 2006.
- **Armijo, Larry.** "Minimization of Functions Having Lipschitz Continuous First Partial Derivatives."  
  *Pacific Journal of Mathematics,* 1966.
- **Wolfe Conditions – Wikipedia:**  
  [https://en.wikipedia.org/wiki/Wolfe_conditions](https://en.wikipedia.org/wiki/Wolfe_conditions)
- **Backtracking Line Search – Wikipedia:**  
  [https://en.wikipedia.org/wiki/Backtracking_line_search](https://en.wikipedia.org/wiki/Backtracking_line_search)
- **JSOSolvers.jl and Optimization.jl:**  
  [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl)  
  [Optimization.jl Documentation](https://docs.sciml.ai/Optimization/)

Additionally, explore related lecture notes on algorithms such as MINRES (for linear systems) and dedicated notes on line search strategies.

---

# Conclusion

In these lecture notes, we have discussed two primary strategies for step size selection in unconstrained optimization:

- **Line Search Methods** rely on finding an appropriate $\alpha$ along a fixed search direction using conditions such as Armijo and Wolfe conditions.  
- **Trust Region Methods** build a local model of the objective and restrict the step to lie within a region where that model is trustworthy.

Both strategies have their theoretical merits and practical trade-offs. Modern optimization solvers often integrate these ideas (or even hybrid methods) to achieve robust and efficient performance.

These notes include examples in both Julia and Python to help illustrate the concepts. For further exploration, links to related topics and additional references are provided.

