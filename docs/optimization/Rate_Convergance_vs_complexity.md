**Overview**

In iterative algorithms—particularly those for solving equations or optimization problems—we care about two related but distinct measures of performance:

1. **Complexity**: How many computational resources (time, flops, memory) does the algorithm use as a function of problem size or tolerance?
2. **Rate of convergence**: How quickly does the sequence of iterates $\{x_k\}$ produced by the algorithm approach the exact solution $x^*$?

We’ll first define and illustrate these notions in general.  Then we’ll work through three concrete methods—Newton’s method, a (local) trust-region method, and the MINRES Krylov solver—showing how to establish their convergence rates and bounds.

---

## 1. Complexity vs. Convergence Rate

### 1.1 Complexity

* **Time complexity** (iterative‐solver perspective): If an algorithm produces an iterate $x_k$ that satisfies $\|x_k - x^*\|\le \varepsilon$, define

  $$
    N(\varepsilon)\;=\;\min\{\,k:\|x_k - x^*\|\le\varepsilon\}.
  $$

  We ask: How does $N(\varepsilon)$ scale as $\varepsilon\to0$?  Typical answers:

  * **Polynomial**: $N(\varepsilon)=\mathcal O(\varepsilon^{-p})$.
  * **Logarithmic**: $N(\varepsilon)=\mathcal O(\ln(1/\varepsilon))$.
  * **Power‐log**: e.g.\ $\mathcal O(\ln\ln(1/\varepsilon))$, etc.

* **Computational cost**: Multiply $N(\varepsilon)$ by the cost per iteration (e.g.\ cost to factor or apply a matrix, line‐search cost, etc.).

### 1.2 Rate of Convergence

Let $\{x_k\}$ be a sequence that converges to $x^*$.  We classify:

1. **Linear convergence**: there exists $0<\rho<1$ and $C>0$ such that

   $$
     \|x_{k+1}-x^*\|\;\le\; C\,\rho^k\,.
   $$

   Equivalently, eventually

   $$
     \|x_{k+1}-x^*\|\;\le\;\rho\,\|x_k - x^*\|\,.  
   $$

2. **Superlinear convergence**:

   $$
     \lim_{k\to\infty}\frac{\|x_{k+1}-x^*\|}{\|x_k - x^*\|}=0.
   $$

   Means error ratio tends to zero.

3. **Quadratic convergence**: there is $M>0$ such that, for all sufficiently large $k$,

   $$
     \|x_{k+1}-x^*\|\;\le\;M\,\|x_k - x^*\|^2.
   $$

4. **Order-$p$ convergence**: if

   $$
     \|x_{k+1}-x^*\|\;\le\;C\,\|x_k - x^*\|^p,
   $$

   for some $p>1$, we say convergence is of order $p$.

---

## 2. How to Prove a Rate and Find Bounds

1. **Model the error recurrence.**  Write the error $e_k = x_k - x^*$ and derive an inequality of the form

   $$
     \|e_{k+1}\|\;\le\;M\,\|e_k\|^p,
   $$

   valid in a neighborhood of $x^*$.  This often uses Taylor expansions or spectral bounds.

2. **Identify constants.**  Carefully bound higher‐order terms to find explicit $M$ and radius of convergence.

3. **Conclude order.**  If $p=1$ with a constant factor $<1$, linear; if $p=2$, quadratic; in general order-$p$.

4. **Translate to iteration count.**  Solve $\|e_k\|\approx C(\|e_0\|)^{p^k}$ or invert the recurrence to get $k$ in terms of tolerance $\varepsilon$.

---

## 3. Example 1: Newton’s Method for Scalar Root-Finding

**Problem.**  Solve $f(x)=0$, with $f:\mathbb R\to\mathbb R$, $f\in C^2$, and assume $f'(x^*)\neq0$.

**Iteration**

$$
  x_{k+1} \;=\; x_k \;-\;\frac{f(x_k)}{f'(x_k)}.
$$

**Error analysis.**  Let $e_k=x_k-x^*$.  Taylor‐expand $f$ around $x^*$:

$$
  f(x_k) = f'(x^*)\,e_k + \tfrac12f''(\xi_k)e_k^2,
  \quad
  f'(x_k)=f'(x^*)+\tfrac12f''(\eta_k)e_k.
$$

Then one shows (for small $e_k$)

$$
  e_{k+1}
  = -\,\frac{ \tfrac12f''(\xi_k)e_k^2 }{f'(x^*) + O(e_k)}
  = -\,\frac{f''(\xi_k)}{2 f'(x^*)} \,e_k^2 + O(e_k^3).
$$

Hence, there exists $M$ so that

$$
  |e_{k+1}|\;\le\;M\,|e_k|^2,
$$

i.e.\ **quadratic convergence**.  Convergence is local: requires $x_0$ sufficiently near $x^*$.  One can bound the region of convergence by ensuring $\|e_k\|$ stays small enough that denominators don’t vanish.

**Iteration count.**

$$
  |e_k|\;\approx\;M^{\,2^k-1}\,|e_0|^{2^k},
$$

so to achieve $|e_k|\le\varepsilon$ one needs only
$\displaystyle k\;\approx\;\log_2\!\bigl(\log_{1/|e_0|}(1/\varepsilon)\bigr)$,
which is extremely fast.

---

## 4. Example 2: (Local) Trust-Region Method for Optimization

**Problem.**  Minimize $f:\mathbb R^n\to\mathbb R$, twice continuously differentiable, with $\nabla^2 f(x^*)$ positive definite at a local minimizer $x^*$.

**Algorithm sketch.**  At iterate $x_k$, build quadratic model

$$
  m_k(s) = f(x_k) + \nabla f(x_k)^Ts + \tfrac12 s^T \nabla^2 f(x_k)\,s,
$$

and solve
$\min_{\|s\|\le\Delta_k} m_k(s)$
to obtain step $s_k$.  Update $x_{k+1}=x_k+s_k$, adjust $\Delta_k$ by trust‐region ratio.

**Local convergence rate.**  Under standard assumptions (Lipschitz Hessian near $x^*$, initial $\Delta_0$ small, steps accepted), one shows that once $x_k$ is close enough:

* The model is a good approximation:
  $\|s_k - (-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k))\| = O(\|\nabla f(x_k)\|)$.
* Since Newton’s step has quadratic convergence, the trust-region steps inherit **quadratic convergence**:

  $$
    \|x_{k+1}-x^*\|\;\le\;C\,\|x_k-x^*\|^2,
  $$

  for some $C$, once $k$ is large.

One proves the bound by combining the model‐error (third‐order Taylor terms) and the fact that for small gradient, the trust‐region constraint is inactive (so one takes the full Newton step).

---

## 5. Example 3: MINRES for Symmetric Linear Systems

**Problem.**  Solve $A x = b$ with $A=A^T$ (possibly indefinite), using MINRES.

**Convergence theory (simplified).**  MINRES finds $x_k$ in the $k$th Krylov subspace $\mathcal K_k(A,r_0)$ minimizing the residual norm $\|b - A x_k\|$.  One can show (via polynomial approximation theory) that

$$
  \frac{\|r_k\|}{\|r_0\|} 
  = \min_{p\in \mathcal P_k,\,p(0)=1} \max_{\lambda\in\mathrm{spec}(A)} |p(\lambda)|,
$$

where $\mathcal P_k$ are polynomials of degree $\le k$.  If $A$ is positive definite with spectrum in $[\lambda_{\min},\lambda_{\max}]$, then the optimal bound uses Chebyshev polynomials to yield **linear** (actually geometric) convergence:

$$
  \frac{\|r_k\|}{\|r_0\|} 
  \;\le\;2\!\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k,
  \quad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}.
$$

Thus MINRES converges linearly with factor $\rho = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ less than 1.  If $A$ has both positive and negative eigenvalues, a similar bound holds on the union of intervals, yielding the same asymptotic rate in $\sqrt{\kappa}$, where $\kappa$ is the ratio of largest to smallest magnitude eigenvalues.

---

## 6. Summary

| Method          | Convergence Rate   | Key bound                                                              |
| --------------- | ------------------ | ---------------------------------------------------------------------- |
| Newton (scalar) | Quadratic ($p=2$)  | $\|e_{k+1}\|\le M\|e_k\|^2$                                            |
| Trust‐region    | Locally quadratic  | $\|x_{k+1}-x^*\|\le C\|x_k-x^*\|^2$                                    |
| MINRES          | Linear (geometric) | $\|r_k\|\le2\bigl(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\bigr)^k\|r_0\|$ |

* **Quadratic methods** (Newton, trust‐region) yield extremely rapid local convergence at the cost of per‐iteration Hessian factorization or model solution.
* **Krylov methods** (MINRES) are matrix‐vector based, cheap per iteration, but only **linear**—though with a condition‐number‐dependent rate.

---

### Take‐Home Messages for Undergraduates

1. **Distinguish** complexity (iterations × cost per iteration) from convergence rate (how error shrinks each iteration).
2. **Classify** convergence by comparing $\|e_{k+1}\|$ to $\|e_k\|$ via constants and exponents.
3. **Use Taylor expansions** or polynomial‐approximation arguments to derive the error recurrence.
4. **Translate** local error recurrences into global statements about $N(\varepsilon)$ vs.\ $\varepsilon$.
5. **Balance** high per‐step cost and fewer steps (Newton/trust‐region) against cheap steps and more iterations (Krylov/CG/MINRES).
