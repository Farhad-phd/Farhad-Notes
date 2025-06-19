## Convergence and Complexity of Newton-MR

*Lecture Notes*

### 1. Notation and Main Assumptions

* **Objective**: Minimize a nonconvex smooth function $f:\mathbb R^n\to\mathbb R$.
* **Gradient and Hessian**:

  * $g_k = \nabla f(x_k)$
  * $H_k = \nabla^2 f(x_k)$

**Assumptions:**

1. **Smoothness**: $f$ is twice continuously differentiable.
2. **Lipschitz Hessian**: There exists $L > 0$ such that for all $x,y$,

   $$
     \|\nabla^2 f(x) - \nabla^2 f(y)\| \le L\,\|x - y\|.
   $$
3. **Bounded Hessian**: There exists $H_{\max}$ such that for all iterates,

   $$
     \|H_k\| \le H_{\max}.
   $$

### 2. Algorithm Overview (Newton-MR)

At each iterate $x_k$:

1. **Krylov Subproblem**: Call MINRES on $H_k s = -g_k$, tracking:

   * **Case A (Inexact Newton):** returns $s_k$ with residual
     $\|H_k s_k + g_k\| \le \eta\,\|g_k\|$, $0<\eta<1$.
   * **Case B (Negative Curvature):** finds $p_k$ with $p_k^T H_k p_k < 0$.
2. **Step Selection:**

   * If Case A, set $s = s_k$.
   * If Case B, set $s = \alpha\,p_k$ for a suitable $\alpha>0$.
3. **Line-Search**: Choose $\alpha_k$ satisfying sufficient decrease.
4. **Update**: $x_{k+1} = x_k + \alpha_k s.$

### 3. Descent Lemmas

#### 3.1 Case A: Inexact Newton Step

Using Taylor expansion with remainder (Lipschitz Hessian):

$$
  f(x_k + s_k) \le f(x_k) + g_k^T s_k + \tfrac12 s_k^T H_k s_k + \tfrac{L}{6}\|s_k\|^3.
$$

Define $r_k = H_k s_k + g_k$, $\|r_k\| \le \eta\|g_k\|$.  Then:

$$
  \Delta f_k = f(x_k)-f(x_k+s_k)
  \ge \tfrac12 s_k^T H_k s_k - \|s_k\|\|r_k\| - \tfrac{L}{6}\|s_k\|^3.
$$

Bounding eigenvalues and choosing $\eta$ small yields:

$$
  \Delta f_k \ge c_1\,\frac{\|g_k\|^{3/2}}{\sqrt{H_{\max}}},
$$

for some constant $c_1>0$.

#### 3.2 Case B: Negative-Curvature Step

Suppose $\mu_k = -p_k^T H_k p_k > 0, \|p_k\|=1$.  For step $s=\alpha p_k$ and optimal $\alpha$, the cubic model gives:

$$
  \Delta f_k \ge \frac{2}{3\sqrt{L}}\,\mu_k^{3/2}.
$$

Thus any negative-curvature direction yields $\Omega(\mu_k^{3/2})$ decrease.

### 4. Complexity Analysis

Let $f^* = \inf_x f(x)$.  Summing $\Delta f_k$ over $K$ iterations,

$$
  f(x_0) - f^* \ge \sum_{k=0}^{K-1} \Delta f_k.
$$

#### 4.1 First-Order Stationarity (Gradient)

Goal: find $\min_k \|g_k\| \le \varepsilon$.

* Each Case A step with $\|g_k\|>\varepsilon$ decreases $f$ by at least $c_1\,\varepsilon^{3/2}/\sqrt{H_{\max}}$.
* Hence at most $\tfrac{f(x_0)-f^*}{c_1}\,(H_{\max})^{1/2}\,\varepsilon^{-3/2}$ such steps.
* Negative-curvature steps only help further.

**Iteration Complexity**: $O(\varepsilon^{-3/2}).$

#### 4.2 Second-Order Stationarity (Hessian)

Goal: ensure $\|g_k\|\le \varepsilon$ *and* smallest eigenvalue $\lambda_{\min}(H_k) \ge -\varepsilon$.

* Negative-curvature steps (Case B) with $\mu_k>\varepsilon$ each yield at least $C_2\,\varepsilon^{3/2}$ decrease.
* Counting both gradient-large and curvature-large phases gives $O(\varepsilon^{-3})$ total iterations.

**Iteration Complexity**: $O(\varepsilon^{-3}).$

### 5. Per-Iteration Cost and Overall Complexity

* **Per Iteration**: one MINRES call (cost ≈ # of Hessian-vector products).  MINRES both solves and detects curvature.
* **Total Hessian-Vector Products**: proportional to iteration count.

| Guarantee            | Iteration Count         | Work per Iteration       |
| -------------------- | ----------------------- | ------------------------ |
| 1st-Order Stationary | $O(\varepsilon^{-3/2})$ | MINRES (\~matrix–vector) |
| 2nd-Order Stationary | $O(\varepsilon^{-3})$   | MINRES (\~matrix–vector) |

### 6. Summary

Newton-MR unifies Newton and negative-curvature steps via a single MINRES subsolver, achieving:

* **Fast local descent**: $O(\|g\|^{3/2})$ or $O(\mu^{3/2})$ per iteration.
* **Optimal worst-case rates**: matching cubic-regularization and trust-region methods, but simpler to implement.

These notes provide the key inequalities and counting arguments you need to understand the convergence and complexity proofs of Newton-MR.
