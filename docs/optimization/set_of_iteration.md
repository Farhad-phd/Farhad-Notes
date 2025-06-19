**Note on Splitting Iterations into $\mathcal{T}_k^\tau$ and $\mathcal{W}_k^\tau$**

---

### 1. Definitions

Let $k_0$ be the initial iteration index (often $1$), and $k$ the current iteration.

1. **Successful vs. Unsuccessful Iterations**

   * $\displaystyle \mathcal{S}   = \{\,i\in\mathbb{N} \mid \rho_i \ge \eta_1\}$
     All **successful** iterations (those with ratio $\rho_i$ above threshold $\eta_1$).
   * $\displaystyle \mathcal{U}   = \{\,i\in\mathbb{N} \mid \rho_i < \eta_1\}$
     All **unsuccessful** iterations.

2. **Iterations up to $k$**

   * $\displaystyle \mathcal{S}_k = \{\,i\in\mathcal{S} : i \le k\}$
     Successful iterations up to and including $k$.
   * $\displaystyle \mathcal{U}_k = \{\,i\in\mathcal{U} : i \le k\}$
     Unsuccessful iterations up to and including $k$.

3. **Cardinality**
   $\lvert \mathcal{S}_j\rvert$ denotes the **number** of elements in the set $\mathcal{S}_j$, i.e.

   $$
     \lvert\mathcal{S}_j\rvert \;=\;\#\{\,i\le j : i\in\mathcal{S}\}.
   $$

4. **Budget Threshold**
   For a parameter $\tau>0$, define at each iteration $j$ the “budget”

   $$
     B_j \;=\;\tau\,\lvert\mathcal{S}_j\rvert.
   $$

5. **Splitting**
   For $j = k_0,\,k_0+1,\,\dots,k$, partition into two sets:

   $$
   \begin{aligned}
     \mathcal{T}_k^\tau 
       &= \bigl\{\,j \mid j_0\le j \le k,\;j \le B_j \bigr\}, 
       &&\text{(“tame” / on‐budget)}\\
     \mathcal{W}_k^\tau 
       &= \bigl\{\,j \mid j_0\le j \le k,\;j > B_j \bigr\}, 
       &&\text{(“wild” / over‐budget)}
   \end{aligned}
   $$

---

### 2. Toy Example

* **Setup**

  * $k_0 = 1,\;k = 6$
  * Success threshold $\eta_1$ yields successes at iterations $\{1,3,4,6\}$.
  * Thus $\mathcal{S} = \{1,3,4,6\}$, $\mathcal{U} = \{2,5\}$.
  * Choose $\tau = 1.5$.

* **Compute for each $j=1,\dots,6$:**

  1. $\lvert\mathcal{S}_j\rvert$
  2. Budget $B_j = \tau\,\lvert\mathcal{S}_j\rvert$
  3. Check if $j \le B_j$

| $j$ | $\lvert\mathcal S_j\rvert$ | $B_j = \tau\,\lvert\mathcal S_j\rvert$ | $j \le B_j$? | Classification       |
| :-: | :------------------------: | :------------------------------------: | :----------: | :------------------- |
|  1  |              1             |                   1.5                  |    ✔️ Yes    | $\mathcal T_6^{1.5}$ |
|  2  |              1             |                   1.5                  |     ❌ No     | $\mathcal W_6^{1.5}$ |
|  3  |              2             |                   3.0                  |    ✔️ Yes    | $\mathcal T_6^{1.5}$ |
|  4  |              3             |                   4.5                  |    ✔️ Yes    | $\mathcal T_6^{1.5}$ |
|  5  |              3             |                   4.5                  |     ❌ No     | $\mathcal W_6^{1.5}$ |
|  6  |              4             |                   6.0                  |    ✔️ Yes    | $\mathcal T_6^{1.5}$ |

* **Resulting Sets**

  $$
    \mathcal T_6^{1.5} = \{1,3,4,6\}, 
    \quad
    \mathcal W_6^{1.5} = \{2,5\}.
  $$

---

### 3. Why This Matters

* **Bounding Iterations.** By separating on-budget vs. over-budget steps, one can show there cannot be too many “wild” iterations without generating more successes (i.e.\ growing $\lvert\mathcal S_j\rvert$) or violating some global bound.
* **Complexity Analysis.** This partition is key in deriving worst-case iteration counts for adaptive algorithms (trust-region, line-search, etc.), ensuring convergence within a finite number of steps.
