# Trading Fingerprints: Bayesian Shrinkage and CLR (Trade-Count Case)

A reference for turning a client's category shares into a stable, comparable feature vector. Core pipeline: shrink the shares toward the population to control noise, then apply the centered log-ratio (CLR) transform so ordinary Euclidean distance behaves correctly on compositional data. Section 6 extends this to **two weightings** — count-weighted ("where attention goes") and dollar-weighted ("where money goes") — and shows how to build the gap between them as its own feature.

---

## 1. Setup and notation

Fix a category universe up front (e.g. asset class, product class, sector, currency), each with an explicit **"other/unknown"** bin so the categories are **mutually exclusive and exhaustive**. This is what guarantees the counts partition cleanly and every client's vector has the same length and meaning.

For a given client and a given categorical dimension with $D$ categories:

| Symbol | Meaning |
|--------|---------|
| $c_i$ | number of the client's trades in category $i$ |
| $n = \sum_i c_i$ | client's total trade count (over this dimension) |
| $p_i = c_i / n$ | client's raw **count-weighted** share in category $i$ ("where attention goes") |
| $w_i$ | client's raw **dollar-weighted** share in category $i$ ("where money goes") |
| $m^{(c)}_i$ | population average **count-weighted** share, with $\sum_i m^{(c)}_i = 1$ |
| $m^{(\$)}_i$ | population average **dollar-weighted** share, with $\sum_i m^{(\$)}_i = 1$ |
| $\alpha$ | pseudocount, in **trades** (a modest value, e.g. 5–20) |

Where a single population mean is written $m_i$ below, it denotes whichever of $m^{(c)}_i$ or $m^{(\$)}_i$ matches the share being shrunk.

---

## 2. Bayesian shrinkage

### The problem

Raw shares treat all clients as equally reliable. A client with 4 trades at a 75% share and a client with 4,000 trades at a 75% share get the same number, but the first is mostly noise (one flipped trade moves it to 50%) and the second is trustworthy. Shrinkage pulls each estimate toward the population average by an amount that depends on how much data the client has.

### Count form

$$
\hat{p}_i = \frac{c_i + \alpha\, m_i}{n + \alpha}
$$

Interpretation: add $\alpha$ imaginary "pseudo-trades" distributed according to the population norm $m$, then recompute the share. Small $n$ → the pseudo-trades dominate → estimate sits near the population average. Large $n$ → the pseudo-trades are a rounding error → estimate stays at the client's own value. This is a Dirichlet (Beta, in the two-category case) prior.

### Equivalent convex-combination form

Dividing numerator and denominator by $n$:

$$
\hat{p}_i = \lambda\, p_i + (1 - \lambda)\, m_i,
\qquad
\lambda = \frac{n}{n + \alpha}
$$

Same object, two views: the count form makes the "add pseudo-trades" story concrete; the convex form makes the mixing weight $\lambda \in [0,1]$ and the sum-preservation explicit.

### Choosing $\alpha$

$\alpha$ sets how many trades of evidence are needed before the client's own data outweighs the population prior. Set it by hand (5–20 is often reasonable) or estimate it from the data: more genuine client-to-client variation relative to within-client noise → trust individuals sooner → smaller $\alpha$.

---

## 3. Shrinkage preserves the constant sum

The shrunk shares still sum to 1, exactly, for any $\alpha$ and any client:

$$
\sum_i \hat{p}_i
= \sum_i \frac{c_i + \alpha m_i}{n + \alpha}
= \frac{\sum_i c_i + \alpha \sum_i m_i}{n + \alpha}
= \frac{n + \alpha \cdot 1}{n + \alpha}
= 1
$$

This relies on two facts:

- $\sum_i c_i = n$ — the category counts partition the client's trades. **This is exactly why the "other/unknown" bin matters:** without it, trades outside the defined categories go uncounted and $\sum_i c_i < n$.
- $\sum_i m_i = 1$ — the population average shares are themselves a composition.

**The one variant that breaks it:** using a **category-specific** pseudocount $\alpha_i$ (equivalently a per-category $\lambda_i$). Then $\lambda$ can no longer be factored out of the sum and $\sum_i \hat{p}_i \neq 1$ in general. A single $\alpha$ per client — applied uniformly across categories — keeps the sum at 1. If you ever go per-category, renormalize afterward.

### A useful side effect: no zeros

Because you always add $\alpha m_i > 0$, no shrunk share is ever exactly 0. This makes the output **CLR-safe** (see below), so shrinkage does double duty: noise control **and** zero-replacement in one step.

---

## 4. Compositional data and why Euclidean distance misleads

The shares $(\hat{p}_1, \dots, \hat{p}_D)$ form a **composition**: parts constrained to sum to a constant, carrying information only in their *relative* proportions. Two issues make raw Euclidean distance on them distorted:

1. **The constraint creates spurious structure.** The parts are not free to move independently — if one share rises, another must fall. The data lives on a $(D-1)$-dimensional simplex, not in free $D$-space, and Euclidean distance cuts across a surface the data can never occupy.

2. **Equal absolute differences aren't equally meaningful.** A move from 1% → 2% and a move from 50% → 51% are both a Euclidean gap of $0.01$, but the first *doubles* the allocation while the second is a trivial nudge. The meaning lives in **ratios**, which Euclidean distance ignores.

---

## 5. Centered log-ratio (CLR) transform

CLR fixes both problems by moving to log-ratio space. For a composition $x = (x_1, \dots, x_D)$:

$$
\operatorname{CLR}(x_i) = \ln\!\left(\frac{x_i}{g(x)}\right),
\qquad
g(x) = \left(\prod_{j=1}^{D} x_j\right)^{1/D}
$$

where $g(x)$ is the **geometric mean** of the client's shares. Each part is expressed as its ratio to the client's "typical" share, then logged. The **ratio** makes differences relative (1%→2% and 50%→51% are now treated by proportional change); the **log** makes them symmetric (doubling and halving are equal and opposite). After CLR, ordinary **Euclidean distance behaves sensibly**.

Feed the shrunk shares $\hat{p}$ (guaranteed positive) directly into CLR:

$$
\text{fingerprint block} = \big(\operatorname{CLR}(\hat{p}_1), \dots, \operatorname{CLR}(\hat{p}_D)\big)
$$

### CLR is scale-invariant

Multiplying every share by a positive constant $c$ leaves CLR unchanged, because $c$ cancels against the geometric mean:

$$
\operatorname{CLR}(c\,x_i)
= \ln\frac{c\,x_i}{g(c\,x)}
= \ln\frac{c\,x_i}{c\,g(x)}
= \ln\frac{x_i}{g(x)}
= \operatorname{CLR}(x_i)
$$

**Consequence:** CLR does not care whether the input sums to 1, to 0.99, or to 100. The constant-sum property (Section 3) is a nice safety net, but it is *only* needed if you feed **raw shares** into something that assumes a composition — plain Euclidean distance on shares, stacked-bar charts, or a probability-expecting model. Once CLR sits in the pipeline before any distance is computed, exact normalization is moot.

> **Note on the CLR covariance singularity.** Because $\sum_i \operatorname{CLR}(x_i) = 0$ by construction, the CLR vector lies on a hyperplane and its covariance matrix is singular. This is harmless for distance and PCA, but if a downstream model requires full rank, use the isometric log-ratio (ILR) transform instead, which maps to $D-1$ unconstrained coordinates.

---

## 6. The attention–money gap as a feature

Two weightings capture different behavior: the **count-weighted** share $p$ ("where attention goes") and the **dollar-weighted** share $w$ ("where money goes"). The discrepancy between them is often the behavioral signal of interest — it distinguishes a client who does *many small trades* in one category but *a few large ones* in another. This section covers how to build that gap feature correctly.

### Keep two of three, never all three

The three quantities — $p$, $w$, and their gap — are **deterministically dependent**: any two determine the third. Keeping all three is not just wasteful but harmful: the exact linear redundancy gets **double-counted** in Euclidean distance and inflates its own eigenvalue in PCA. **Keep exactly two.** Which two is a modeling choice:

- **$p$ + $w$** — treats attention and money symmetrically; the gap is present implicitly but cannot be weighted as its own concept.
- **$p$ + gap** (*recommended when the gap is the point*) — a "base profile" plus "money's deviation from it." Makes the discrepancy an explicit, separately **weightable** block, so you can up- or down-weight attention-vs-money divergence as a knob.

### Compute the gap *after* shrinkage, not before

Shrink both shares first, then form the gap. Two reasons:

1. **Noise.** The gap is a difference of two estimated quantities, so it is noisier than either alone. For a low-trade client, $p$ and $w$ are each noisy and their raw difference is noise-on-noise.
2. **Correct prior.** Shrink each share toward **its own** population weighting — $p$ toward $m^{(c)}$, $w$ toward $m^{(\$)}$. Then for a low-data client both collapse toward their respective population means, so the gap collapses toward the **population-level** attention–money gap. That is exactly the right default: with little evidence, assume this client has the *typical* discrepancy, not a wild personal one. Computing the gap pre-shrinkage discards this.

### Define the gap multiplicatively: it *is* the difference of CLRs

Do **not** apply CLR to the additive difference $w_i - p_i$. That difference has signed entries and sums to zero,

$$
\sum_i (w_i - p_i) = \sum_i w_i - \sum_i p_i = 1 - 1 = 0,
$$

so it is not a composition (not on the simplex) but a **contrast vector** in the tangent space; $\ln(\cdot)$ and the geometric mean do not apply to it.

Instead, use the natural compositional difference — the ratio $w_i / p_i$ — whose centered log-ratio is **exactly the difference of the two CLR vectors**:

$$
\text{gap}_i
\;=\; \operatorname{CLR}(w)_i - \operatorname{CLR}(p)_i
\;=\; \ln\frac{w_i}{g(w)} - \ln\frac{p_i}{g(p)}
\;=\; \ln\frac{w_i / p_i}{g(w)/g(p)}
$$

This is the CLR of the compositional perturbation $w \ominus p$. It is **already real-valued** — CLR coordinates are logs, so their difference is naturally signed (positive and negative), needing no further transform — and it sums to zero like any CLR vector, which is expected and fine.

Interpretation: $\text{gap}_i > 0$ means **dollars over-weight** category $i$ relative to attention — the client's *few big trades* cluster there; $\text{gap}_i < 0$ means **attention over-weights** it — *many small tickets* there.

### Consequence for redundancy

Because the log-ratio gap is *exactly* $\operatorname{CLR}(w) - \operatorname{CLR}(p)$, keeping all three CLR blocks ($\operatorname{CLR}(p)$, $\operatorname{CLR}(w)$, gap) is **exact linear redundancy** — unambiguously to be avoided. The clean recommendation: keep **$\operatorname{CLR}(p)$ plus the gap $\operatorname{CLR}(w) - \operatorname{CLR}(p)$** — interpretable, non-redundant, each independently weightable.

### If you insist on the additive gap $w - p$

It is already a signed real vector, so it needs **no** log-type transform — just **z-score each component** across clients. If it is heavy-tailed (a few clients with extreme discrepancies), apply an **inverse hyperbolic sine** (signed log) first:

$$
\operatorname{asinh}(x/s) = \ln\!\left(\frac{x}{s} + \sqrt{(x/s)^2 + 1}\right)
$$

which is linear near zero, logarithmic in the tails, odd (preserves sign), and defined for all reals. Caveat: asinh treats components independently and ignores the compositional geometry, so it is a pragmatic fix, not a principled one — the price of going additive. The log-ratio gap above is preferred because it stays inside the same Aitchison geometry as the rest of the pipeline. (ILR works identically: apply it to $w$ and $p$ separately, then difference — with the bonus of avoiding the sum-to-zero singularity if a downstream model needs full rank.)

### The gap and effective sample size ($n_{\text{eff}}$) are in tension

The Kish effective sample size $n_{\text{eff}} = (\sum_j v_j)^2 / \sum_j v_j^2$ is an optional refinement that shrinks the **dollar** block harder when one whale trade dominates a client's volume. But it and the gap treat whale behavior oppositely, so they pull against each other:

- The **gap** treats whales as **signal** — "big money in X, small tickets in Y" is exactly the feature it exists to capture.
- **$n_{\text{eff}}$** treats whales as **unreliability** — a whale-dominated composition rests on essentially one observation, so it shrinks $\operatorname{CLR}(w)$ toward the population dollar mean, which drags the gap toward the population-typical gap and **dampens the very divergence the gap is meant to record.**

So if you have committed to the gap as a feature, $n_{\text{eff}}$ is not merely unnecessary — it is counterproductive. Two further points make the default clear:

1. **Ordinary count-based shrinkage already covers the small-sample case.** A low-trade client shrinks both $p$ and $w$ toward their population means with the same small $\lambda$, so the gap automatically collapses to the population gap. $n_{\text{eff}}$ is not needed for that.
2. **What $n_{\text{eff}}$ uniquely catches** is the client with *many* trades but whale-dominated dollars (e.g. 100 trades, one at 95% of volume), where $n_{\text{trades}}$ says "trustworthy" but the dollar composition really rests on one trade.

Whether that last case is signal or noise — a recurring habit or a one-off — cannot be settled by the gap or $n_{\text{eff}}$ alone; both only see the aggregated window. The **temporal stability check** decides it: split the client's history in half and test whether the whale-driven gap replicates. If it does, it is a trait — keep it, leave $n_{\text{eff}}$ off. If it does not, it is noise — then reach for $n_{\text{eff}}$ (or winsorizing trade sizes).

**Default: $n_{\text{eff}}$ off**, precisely because the gap is the signal and $n_{\text{eff}}$ erodes it. Switch it on only if validation shows the gap is unstable. Note also that the evidence count governing shrinkage — whether $n_{\text{trades}}$ or $n_{\text{eff}}$ — is always in **trade-count units**, never dollar volume; feeding dollar amount into that slot reintroduces the magnitude mismatch that the convex-combination form exists to avoid.

---

## 7. Per-client computation order

1. **Aggregate** per category, including the "other/unknown" bin: trade counts $c_i$ (with $n = \sum_i c_i$) and dollar volumes → raw shares $p_i$ and $w_i$.
2. **Shrink** each share toward **its own** population weighting:
   $\displaystyle \hat{p}_i = \frac{p_i \cdot n + \alpha\, m^{(c)}_i}{n + \alpha}$ toward $m^{(c)}$, and $\hat{w}_i$ toward $m^{(\$)}$ by the equivalent convex form $\hat{w}_i = \lambda w_i + (1-\lambda) m^{(\$)}_i$. Both sum to 1 and have no zeros.
3. **CLR-transform** each: $\displaystyle \operatorname{CLR}(\hat{p})_i = \ln\!\big(\hat{p}_i / g(\hat{p})\big)$, likewise $\operatorname{CLR}(\hat{w})$.
4. **Form the gap block**: $\text{gap}_i = \operatorname{CLR}(\hat{w})_i - \operatorname{CLR}(\hat{p})_i$.
5. **Select two of three** blocks — recommended: $\operatorname{CLR}(\hat{p})$ and the gap. Never keep all three (exact linear redundancy).
6. **Concatenate** the selected blocks across all categorical dimensions into the client's fingerprint vector; optionally apply per-block weights.
7. **Compare** with Euclidean (or cosine) distance; optionally PCA for compression/visualization.

---

## 8. Summary of the guarantees

- The shrinkage estimator $\hat{p}$ (and $\hat{w}$) **sums to exactly 1** for any $\alpha$, provided categories are exhaustive (the "other" bin) and $\alpha$ is a single value across categories.
- Shrinkage **eliminates zeros**, making the output CLR-safe with no separate zero-replacement step.
- **CLR is scale-invariant**, so the constant-sum property is guaranteed but not strictly required once CLR precedes any distance computation.
- The **count-weighted, dollar-weighted, and gap** blocks are mutually determined; keep exactly **two of the three**. The log-ratio gap is exactly $\operatorname{CLR}(\hat{w}) - \operatorname{CLR}(\hat{p})$, so keeping all three is exact linear redundancy.
- Compute the **gap after shrinkage**, shrinking each weighting toward its own population mean, so low-data clients collapse to the population-typical gap.
- The only structural pitfall in shrinkage is a **category-specific $\alpha_i$**, which breaks the sum-to-1 property; renormalize if you go that route.
