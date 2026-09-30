I want to extract trading behavior for each client of a wealth management bank. I have trade data for each client. Each trade has amount traded in USD and in quantity of instrument, trade date, trade currency, buy or sell and various informations regarding the traded financial security. I have asset class, eg equities, bonds and so on. I have product class, eg common stock, option or structured product. For many instruments I also have sector of the traded instrument, eg technology, utilities or similar. 
The goal is to have a numeric representation of the clients trading activity, ie a trading fingerprint. 
Fingerprints should be comparable between clients. 
What approach do you suggest?


The core design decision is to separate *scale* (how much a client trades) from *style* (how they trade), and build the fingerprint mostly out of scale-invariant quantities. Two clients trading $100k and $10M should look similar if they behave similarly. That means expressing most features as **shares and ratios** rather than absolute amounts, so a small retail client and a large one are directly comparable. Below is a concrete pipeline.

## 1. Fix the category universe first

Define the full set of asset classes, product classes, sectors, and currency buckets up front, each with an explicit "other/unknown" bin. This guarantees every client's vector has the same length and that dimension *i* means the same thing for everyone. The sector NAs (common for bonds/FX) become a real "no sector" bucket rather than being dropped — the missingness itself is informative.

## 2. Feature families

**Composition (the bulk of the fingerprint).** For each categorical dimension, compute the client's share of activity: % in equities/bonds/FX/…, % in common stock/options/structured products, % by sector, % by currency. Compute these two ways — weighted by USD amount (where the money goes) and by trade count (where attention goes). The gap between the two weightings is itself a useful signal: someone doing many tiny option trades and a few huge bond trades looks very different under each.

**Directionality.** Buy-share of volume (buy USD / total USD), gross vs net turnover. Distinguishes accumulators from churners.

**Concentration / diversification.** Herfindahl index or Shannon entropy over instruments and over sectors, plus count of distinct instruments/sectors. Captures the diversified buy-and-hold client vs the concentrated punter.

**Complexity / sophistication.** Share of derivatives and structured products; option usage. A rough proxy for how advanced the client is.

**Intensity (scale — include sparingly).** Trades per unit time, average trade size, total volume. These are *not* scale-invariant, so log-transform them and, if you have AUM, normalize by it. Keep only a few, or they'll dominate and your fingerprint will mostly encode wealth.

**Temporal dynamics.** Inter-trade time (mean and coefficient of variation → steady vs bursty), and if you can match buys to sells of the same instrument, an approximate holding period.

## 3. Transforms before you compare

Two subtleties matter here:

- **Compositional data.** Shares live on a simplex with a constant-sum constraint, so raw Euclidean distance on them is distorted. Use a **centered log-ratio (CLR)** transform on each composition block. Because clients have many zero shares, you need a zero-replacement step first (small pseudocount / multiplicative replacement, or a Dirichlet prior — which doubles as smoothing for low-trade clients).
- **Continuous features.** Log-transform the heavy-tailed ones, then z-score across the client population so no single feature dominates the distance purely because of its units.

## 4. From features to a comparable fingerprint

Concatenate the blocks into one standardized vector — that's already a usable fingerprint, and it's interpretable. "Comparable" then means a distance: cosine if you care about profile shape over magnitude, Euclidean on the standardized vector otherwise. You can block-weight (decide how much composition vs. intensity vs. directionality should count) to encode what "similar" should mean.

If you want something more compact or want to discover styles, two options:
- **PCA** on the standardized vector — decorrelates and compresses; good for visualization and clustering.
- **A "trades as text" framing** — treat each instrument/sector/product as a token and each client as a document, apply TF-IDF (which automatically down-weights instruments everyone trades, like large-cap equities), then topic modeling (LDA/NMF). Each client becomes a mixture over latent "trading styles," which is a very natural, interpretable fingerprint.

## 5. Validate that it captures a stable trait

The cleanest test: split each client's history into two time windows and fingerprint each half. A good fingerprint places a client's two halves closer to each other than to other clients (test–retest reliability). Beyond that, check whether clusters map to recognizable personas (income investor, options speculator, FX hedger, diversified holder) and whether the fingerprint predicts something you care about downstream (product uptake, churn, risk flags).

## Main pitfalls

The AUM confound is the big one — decide deliberately whether wealth should be *in* or *out* of the fingerprint, because without normalization it dominates. Low-trade clients are noisy, so apply a minimum-trade threshold or Bayesian shrinkage toward the population average. And normalize activity per unit time, since clients have different history lengths.

If you tell me roughly how many clients and trades you're working with, whether you have AUM/position data, and whether the end use is clustering, similarity search, or a downstream model, I can tighten this into a specific feature list and a recommended distance/embedding choice.



Can you explain a bit more on the following. What is meant by compositional data and the use of Euclidean vs CLR? 
What is meant by Bayesian shrinkage?

Both concepts address the same underlying problem — that the "obvious" way to handle your share and rate features gives distorted answers — but in different places. Let me take them in turn.

## Compositional data and why Euclidean distance misleads

**What it is.** A composition is a set of parts that are constrained to sum to a constant (here, 1 or 100%). Your asset-class shares are a composition: if a client is 60% equities, 30% bonds, 10% FX, those *must* add to 100%. The information is entirely in the *relative* proportions, not the absolute numbers.

The key consequence is that the parts are **not free to move independently**. If equities goes up, something else must go down. This built-in negative correlation is what breaks ordinary distance.

**Why Euclidean distance is distorted.** Two problems:

*Problem 1 — the constraint creates spurious structure.* Euclidean distance treats each dimension as independent, but they can't be. The data doesn't live in ordinary 3D space; it lives on a 2D triangle (the simplex) embedded in it. Distances measured through the ambient space cut across a surface the data can never occupy.

*Problem 2 — equal absolute differences aren't equally meaningful.* This is the intuitive one. Consider the change from 1% to 2% versus 50% to 51%. Euclidean distance calls both a gap of 0.01 — identical. But going from 1% to 2% is a *doubling* of that client's allocation to the sector, while 50% to 51% is a trivial 2% relative nudge. In trading terms, a client who moves from 1% options to 2% options has meaningfully changed their behavior; a client going 50%→51% has not. Euclidean distance is blind to this because it works on absolute differences, whereas the meaning lives in ratios.

**What CLR does.** The centered log-ratio transform fixes both by working in log-ratio space. For a client with shares $(x_1, \dots, x_D)$:

$$\text{CLR}(x_i) = \ln\left(\frac{x_i}{g(x)}\right), \quad g(x) = \left(\prod_j x_j\right)^{1/D}$$

where $g(x)$ is the geometric mean of all the shares. In words: for each part, take its ratio to the client's "typical" share, then take the log. Two things happen. The **ratio** makes it relative — now 1%→2% and 50%→51% are treated according to their proportional change, not their absolute change. The **log** makes ratios symmetric, so doubling and halving are equal-and-opposite moves rather than one being capped at 1 and the other unbounded. After the transform you can use ordinary Euclidean distance on the CLR values and it behaves sensibly.

This is exactly why the **zero-replacement** step I mentioned is mandatory: $\ln(0)$ is undefined, and a client with 0% in some sector is the norm, not the exception. You nudge zeros to a tiny positive value first — which is where the second concept comes in.

## Bayesian shrinkage

**The problem it solves.** Suppose a client has made 4 trades, 3 of them equities. Their measured equity share is 75%. Do you believe that's their true tendency? Not really — with only 4 trades, that number is mostly noise. Flip one trade and it's 50%. Meanwhile a client with 4,000 trades at 75% equities — you believe *that*. Raw shares treat both as equally reliable, which is wrong.

**The idea.** Shrinkage pulls each client's estimate *toward the population average*, by an amount that depends on how much data that client has. Little data → pulled hard toward the average (you don't trust the client's own numbers much). Lots of data → barely moved (their own history speaks for itself).

A simple, common form: instead of

$$\hat{p} = \frac{\text{trades in category}}{\text{total trades}}$$

use

$$\hat{p} = \frac{\text{trades in category} + \alpha \cdot m}{\text{total trades} + \alpha}$$

where $m$ is the population's average share for that category and $\alpha$ is a "pseudocount" — think of it as $\alpha$ imaginary trades drawn from the population norm that you add to everyone. For the 4-trade client with, say, $\alpha = 10$, those 10 pseudo-trades dominate and their estimate sits close to the population average. For the 4,000-trade client, 10 extra trades are a rounding error and their estimate stays at ~75%. Same formula, self-adjusting trust. (This is a Beta/Dirichlet prior, if you want the formal name.)

**Two jobs at once.** Notice this also solves the zero problem for CLR: because you're always adding $\alpha \cdot m > 0$, no share is ever exactly zero. So one step buys you both the noise reduction and the log-safety. That's why I bundled them.

The one judgment call is choosing $\alpha$ — how many trades before you trust a client over the population. You can set it by hand (a modest value like 5–20 is often reasonable) or estimate it from the data by seeing how much genuine client-to-client variation there is relative to within-client noise. More variation between clients → trust individuals sooner → smaller $\alpha$.

The through-line for both: naive formulas (Euclidean distance, raw shares) implicitly assume things your data violates — independence and equal reliability. CLR and shrinkage each restore a correct treatment, and conveniently the fix for one enables the other. Want me to sketch the actual computation order for a client, from raw trades through shrinkage and CLR to a finished vector?



how would bayesian shrinkage look like for the USD weighted shares? the population weighted average is probably orders of magnitude differenrent from the average trade count, i.e. does that still work then? or should one take the log of volumnes first in order to reduce the magnitudes? would the latter  make sense?


You've spotted a real problem, and the fix is cleaner than either rescaling or logging. Let me separate the two things you're asking about, because they're actually distinct.

## Why the naive plug-in breaks

The count formula

$$\hat{p}_i = \frac{c_i + \alpha m_i}{n + \alpha}$$

only works because $\alpha$ and $n$ are in the *same units* — trades. $\alpha$ is "how many pseudo-trades of population-average behavior do I add." If you swap in USD, then $n$ becomes total dollar volume (say \$10M) and $\alpha$ would also have to be in dollars. A fixed $\alpha = 10$ is now ten *dollars* against ten million — utterly negligible, so nothing gets shrunk. To shrink a \$10M client meaningfully you'd need $\alpha$ in the millions, but then that same $\alpha$ would obliterate a \$10k client. There's no single $\alpha$ that works across the population. So your instinct is right: it does not work as-is.

## The deeper reason, which points to the fix

Here's the key realization: **the dollar magnitude was never the right measure of how much to trust a client's composition.** Reliability comes from the *number of trades*, not the dollar total. A client with one \$10M trade has a USD-weighted composition that's 100% concentrated in whatever that single trade was — maximally unreliable, despite the huge dollar figure. A client with 1,000 trades totaling \$10M has a far more trustworthy composition. Same dollars, completely different reliability. So rescaling the dollars doesn't address the real issue.

The fix is to stop feeding dollars into the "sample size" slot at all. Write shrinkage as a **convex combination** of the client's share and the population share, and let the mixing weight be governed by trade count:

$$\hat{w}_i = \lambda\, w_i + (1-\lambda)\, m_i, \qquad \lambda = \frac{n}{n + \alpha}$$

where $w_i$ is the client's USD-weighted share in category $i$, $m_i$ is the population's USD-weighted average share, $n$ is the client's trade count, and $\alpha$ is a pseudocount **in trades** (same modest 5–20 as before). This is algebraically the same Dirichlet shrinkage — just divide the count formula's numerator and denominator by $n$ to see it — but now the thing being mixed (compositions) and the thing governing the mix (evidence count) live in different units, and that's fine. The magnitude mismatch simply disappears because both $w_i$ and $m_i$ are shares in $[0,1]$. You get the same benefit for free: since you always add $(1-\lambda)m_i > 0$, no share is exactly zero, so it's still CLR-safe.

## A refinement that's especially apt for USD weighting

Trade *count* isn't quite the perfect reliability measure for a dollar-weighted share, because a client with 100 trades where one whale is 90% of the volume has a composition dominated by that single event — it should be trusted less than 100 evenly-sized trades. You can capture this with an **effective sample size** (Kish's formula) instead of the raw count:

$$n_{\text{eff}} = \frac{\left(\sum_j v_j\right)^2}{\sum_j v_j^2}$$

where $v_j$ are the individual trade USD amounts. This equals $n$ when all trades are equal size and collapses toward 1 when one trade dominates. Plug $n_{\text{eff}}$ into $\lambda = n_{\text{eff}}/(n_{\text{eff}} + \alpha)$ and whale-dominated compositions automatically get shrunk harder toward the population — which is exactly the behavior you want for a money-weighted feature.

## On taking logs of volumes

This is a legitimate idea, but it answers a *different* question, and I'd caution against reaching for it as the shrinkage fix. Two separate issues are in play:

The magnitude mismatch in shrinkage — that's fully solved by the convex-combination reformulation above. Logging volumes to make the units smaller would be treating the symptom with the wrong tool.

Whether raw dollar-weighting is even the weighting you want — this is the real question logging speaks to. Dollar-weighting lets a few large trades dominate a client's whole profile. Logging individual trade sizes before computing shares compresses that: a \$10M trade counts only ~2.3× a \$1k trade (ratio of their logs) instead of 10,000×. So it's a deliberate choice about *how much a big trade should count* toward "style."

The problem is interpretability. Count-weighted shares mean "where attention goes"; dollar-weighted shares mean "where the money goes." A log-dollar-weighted share is a sum of log-dollars, which isn't a meaningful quantity — it sits in an awkward middle with no clean reading. So I'd only log if you specifically want that whale-compression *and* accept the interpretive murkiness.

If whale dominance is your actual worry, the cleaner levers are: use $n_{\text{eff}}$ (which already down-weights whale-driven compositions during shrinkage), winsorize or cap extreme trade sizes before computing shares, or — my original suggestion — keep both the count-weighted and dollar-weighted versions and treat the *gap between them* as its own feature. That gap directly encodes "this client does many small trades in X but a few huge ones in Y," which is often exactly the behavioral signal you're after, and you get it without giving up interpretability on either end.

So: reformulate shrinkage as a share-space convex combination governed by (effective) trade count, and make the weighting-scheme choice on interpretability grounds rather than as a workaround. Want me to write out the full per-client computation order with $n_{\text{eff}}$ and CLR included?







# Trading Fingerprints: Bayesian Shrinkage and CLR (Trade-Count Case)

A reference for turning a client's **trade-count-weighted category shares** into a stable, comparable feature vector. Two steps: shrink the shares toward the population to control noise, then apply the centered log-ratio (CLR) transform so ordinary Euclidean distance behaves correctly on compositional data.

---

## 1. Setup and notation

Fix a category universe up front (e.g. asset class, product class, sector, currency), each with an explicit **"other/unknown"** bin so the categories are **mutually exclusive and exhaustive**. This is what guarantees the counts partition cleanly and every client's vector has the same length and meaning.

For a given client and a given categorical dimension with $D$ categories:

| Symbol | Meaning |
|--------|---------|
| $c_i$ | number of the client's trades in category $i$ |
| $n = \sum_i c_i$ | client's total trade count (over this dimension) |
| $p_i = c_i / n$ | client's raw count-weighted share in category $i$ |
| $m_i$ | population average share in category $i$, with $\sum_i m_i = 1$ |
| $\alpha$ | pseudocount, in **trades** (a modest value, e.g. 5–20) |

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

## 6. Per-client computation order

1. **Count** trades per category, including the "other/unknown" bin → $c_i$, and $n = \sum_i c_i$.
2. **Shrink** toward the population: $\displaystyle \hat{p}_i = \frac{c_i + \alpha m_i}{n + \alpha}$. Output sums to 1 and has no zeros.
3. **CLR-transform**: $\displaystyle \operatorname{CLR}(\hat{p}_i) = \ln\!\big(\hat{p}_i / g(\hat{p})\big)$.
4. **Concatenate** the CLR blocks from each categorical dimension into the client's fingerprint vector.
5. **Compare** with Euclidean (or cosine) distance; optionally PCA for compression/visualization.

---

## 7. Summary of the guarantees

- The shrinkage estimator $\hat{p}$ **sums to exactly 1** for any $\alpha$, provided categories are exhaustive (the "other" bin) and $\alpha$ is a single value across categories.
- Shrinkage **eliminates zeros**, making the output CLR-safe with no separate zero-replacement step.
- **CLR is scale-invariant**, so the constant-sum property is guaranteed but not strictly required once CLR precedes any distance computation.
- The only structural pitfall is a **category-specific $\alpha_i$**, which breaks the sum-to-1 property; renormalize if you go that route.



is the bayesian shrinkage approach connected to a bayesian naive bayes with dirichlet prior?

Yes — quite directly. They're the *same estimator*; what differs is the prior you plug in and what you do with the result afterward.

## The shared core: Dirichlet–multinomial conjugacy

Both are the posterior mean of a multinomial parameter under a Dirichlet prior. Put a Dirichlet prior with concentration parameters $\alpha_i = \alpha m_i$ on the share vector $p$, observe category counts $c = (c_1, \dots, c_D)$ under a multinomial likelihood, and the posterior is Dirichlet with parameters $\alpha m_i + c_i$. Its mean is

$$
\mathbb{E}[p_i \mid c] = \frac{c_i + \alpha m_i}{\sum_j (c_j + \alpha m_j)} = \frac{c_i + \alpha m_i}{n + \alpha}
$$

using $\sum_j \alpha m_j = \alpha$ and $\sum_j c_j = n$. That's exactly the shrinkage formula. So our estimator *is* the Dirichlet–multinomial posterior mean — nothing more, nothing less.

Naive Bayes with a Dirichlet prior uses the identical object. In multinomial NB the class-conditional feature probabilities are estimated as

$$
\hat{\theta}_{i\mid k} = \frac{N_{ik} + \alpha_i}{N_k + \sum_j \alpha_j}
$$

which is the same posterior mean, applied per class. Laplace smoothing ($\alpha_i = 1$) and Lidstone smoothing are just particular choices of the Dirichlet pseudocounts. So the "add pseudocounts and renormalize" move is one and the same in both settings.

## Where they differ

**The prior mean.** Vanilla NB usually shrinks toward a *symmetric/uniform* prior ($\alpha_i$ constant, so the implied prior mean is $1/D$) — there's no natural "population feature distribution" it's backing off to. Our shrinkage instead sets the prior mean to the **pooled population shares** $m$, estimated from all clients. That makes ours an **empirical Bayes** procedure (data-driven prior) rather than a fixed-prior one. You could make NB empirical-Bayes too, but out of the box it typically isn't.

**The purpose.** This is the bigger difference. NB is a *generative classifier*: it uses those smoothed class-conditional distributions as likelihood factors, multiplies them under the conditional-independence assumption, and applies Bayes' rule over classes to produce $P(\text{class} \mid \text{features})$. The Dirichlet smoothing there is defensive — a single zero probability would otherwise zero out the whole product. We're doing none of that. We use the smoothed distribution as a *descriptive representation* of the client, feed it into CLR, and compare with a distance. No class posterior, no likelihood product, no classification. Same estimator, different job.

## The deeper parallel worth naming

There's a second connection beyond the smoothing formula. We shrink each categorical dimension — asset class, product class, sector, currency — as its *own* Dirichlet, rather than estimating one giant multinomial over the joint (asset × product × sector × …) cells. That factorization into separately-modeled marginals is the same *flavor* of move as the "naive" assumption in Naive Bayes: back off from a joint that's far too sparse to estimate toward lower-dimensional pieces plus smoothing. It's driven by the same curse-of-dimensionality pressure.

It isn't literally NB's assumption — NB asserts features are independent *given the class*, whereas we're just modeling marginals independently at the estimation stage and then letting the distance metric or PCA recover cross-dimension correlations downstream. But the statistical motivation is identical: the full joint has too many cells relative to your trade counts, so you factorize and smooth.

So the honest one-line answer: the shrinkage step is the Dirichlet–multinomial posterior mean, which is exactly what Bayesian/Laplace-smoothed Naive Bayes uses for its class-conditionals — ours differs only in using an empirical-Bayes population prior instead of a symmetric one, and in serving as a descriptive feature rather than a classifier component. And the per-dimension factorization is a genuine (if looser) cousin of the naive independence assumption.




re the above keeping the trade count weighted and the dollar-weighted shares and using the gap, i.e. the difference between them: 

1. when adding the gap between both version as a feature, can i remove one of them? it is somewhat redundant or not?
2. would you calculate the gap before or after applying bayesian shrinkage or  similar?
3. should the centered log ratio also be applied to the gap? since its a difference is might have positive and negative values, i.e. applying a log does not work neither does computing the geometric mean.
4. what would be an alternative to the CLR, which can handle positive and negative values and which makes sense in this context?



Good — these four questions interlock, and the answers to 3 and 4 turn out to make question 1 cleaner too. Let me take them in order, then tie them together.

## 1. Redundancy — can you drop one?

Yes. The three quantities — count-weighted share $p$, dollar-weighted share $w$, and gap — are **deterministically dependent**: any two determine the third. So from an information standpoint one is fully redundant, and keeping all three is genuinely harmful, not just wasteful. Exact linear redundancy means that axis of variation gets **double-counted** in Euclidean distance and inflates its own eigenvalue in PCA. Keep exactly **two of the three**.

Which two is a modeling choice about what you want the distance sensitive to:

- **$p$ + $w$** (two co-equal profiles): treats "where attention goes" and "where money goes" symmetrically. The gap information is present implicitly but you can't weight it as a concept.
- **$p$ + gap** ("base profile" + "money's deviation from it"): my preferred option if the gap is behaviorally the point. It makes the discrepancy an explicit, separately-**weightable** block — you can up- or down-weight "attention-vs-money divergence" as a knob, which you cannot do when it's buried inside $p$ and $w$.

So: drop one, and I'd lean toward keeping the count-weighted profile plus the gap.

## 2. Gap before or after shrinkage?

**After — always.** Two reasons:

The gap is a difference of two estimated quantities, so it's *noisier* than either one alone. For a low-trade client, $p$ and $w$ are each noisy and their difference is noise-on-noise. Shrinking both first means the gap reflects real divergence rather than sampling accident.

More importantly, shrink each share toward **its own** population weighting — $p$ toward the population count-weighted mean $m^{(c)}$, $w$ toward the population dollar-weighted mean $m^{(\$)}$. Then for a low-data client both collapse toward their respective population means, so the gap collapses toward the **population-level** attention–money gap $\big(\text{transform}(m^{(\$)}) - \text{transform}(m^{(c)})\big)$. That's exactly the right prior: with little evidence, assume this client has the *typical* discrepancy, not a wild personal one. Computing the gap pre-shrinkage throws that away.

## 3. Should CLR apply to the gap?

Not to the *additive* gap — and your reasoning is correct. The difference $w_i - p_i$ has signed entries and in fact sums to zero:

$$\sum_i (w_i - p_i) = \sum_i w_i - \sum_i p_i = 1 - 1 = 0$$

It's not a composition (not on the simplex), it's a **contrast vector** in the simplex's tangent space, so $\ln(\cdot)$ and the geometric mean don't apply. CLR is the wrong tool for it.

But there's an elegant fix: **define the gap multiplicatively instead of additively**, and then no extra transform is needed. The natural "difference between two compositions" in this geometry is the ratio $w_i / p_i$, and its centered log-ratio is *exactly the difference of the two CLR vectors*:

$$\text{gap}_i \;=\; \operatorname{CLR}(w)_i - \operatorname{CLR}(p)_i \;=\; \ln\frac{w_i}{g(w)} - \ln\frac{p_i}{g(p)} \;=\; \ln\frac{w_i/p_i}{g(w)/g(p)}$$

This is the CLR of the compositional perturbation $w \ominus p$. It's **already real-valued** — CLR coordinates are logs, so their difference is naturally signed, positive and negative, no further transform required. It sums to zero like any CLR vector, which is expected and fine.

And it's interpretable: $\text{gap}_i > 0$ means dollars over-weight category $i$ relative to attention — the client's *few big trades* cluster there — while $\text{gap}_i < 0$ means attention over-weights it — *many small trades* there. That "which categories do I do a few whales in vs. lots of small tickets in" signal is often exactly the behavioral fingerprint you're chasing.

So: don't CLR the additive gap; instead let the gap **be** the difference of CLRs.

## 4. Alternative for signed values

Framed by question 3, this mostly dissolves — but here's the honest hierarchy:

**Best in this context: don't additively difference at all.** Use $\operatorname{CLR}(w) - \operatorname{CLR}(p)$ from above. It *is* "the log-ratio transform that handles positive and negative values," because logs of ratios are inherently signed. This keeps you inside Aitchison geometry and consistent with the rest of the pipeline. (ILR instead of CLR works identically here — apply it to $w$ and $p$ separately, then difference — and has the bonus of avoiding the sum-to-zero singularity if a downstream model needs full rank.)

**If you insist on the additive gap $w - p$:** it's already a signed real vector, so it needs *no* log-type transform — just **z-score each component** across clients and use it directly. If it's heavy-tailed (a few clients with extreme discrepancies), apply a **signed log / inverse hyperbolic sine** first:

$$\operatorname{asinh}(x/s) = \ln\!\left(\frac{x}{s} + \sqrt{(x/s)^2 + 1}\right)$$

which is linear near zero, logarithmic in the tails, odd (preserves sign), and defined for all reals — the standard robustifier for signed heavy-tailed quantities. The caveat: asinh treats components independently and ignores the compositional geometry, so it's a pragmatic fix, not a principled one. That's the price of going additive.

## Tying it back to question 1

Notice questions 3 and 4 sharpen the redundancy answer. If you keep $\operatorname{CLR}(p)$ and $\operatorname{CLR}(w)$, then the log-ratio gap is *exactly* their linear difference — so keeping all three CLR blocks is now **exact linear redundancy**, unambiguously to be avoided. The clean recommendation lands as: compute both shrunk shares, transform both with CLR (or ILR), and keep **$\operatorname{CLR}(p)$ plus the gap $\operatorname{CLR}(w) - \operatorname{CLR}(p)$** as your two blocks — interpretable, non-redundant, and each independently weightable.

Want me to fold this — the gap definition, the before/after-shrinkage ordering, and the two-blocks-not-three rule — into the reference document?