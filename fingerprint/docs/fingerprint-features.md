# Trading Fingerprint: Feature Reference

The "fingerprint" is a numeric profile built for every client from their raw trade
history. Instead of just looking at how much a client trades, it captures *how*
they trade — what they trade, in which direction, how concentrated or diversified
they are, how sophisticated their product mix is, how active they are, and what
rhythm their trading follows. Each client ends up as one row of features that can
be compared, clustered, or explained.

The features fall into six blocks. The first block (composition) is repeated for
every categorical dimension available in the trade data — in the current setup
that's **asset class**, **product class**, and **sector** — so the same feature
logic is reused across all three without being hard-coded to any one of them.

## Two tables: `fp` and `fp_raw`

`build_fingerprint(...)` returns a `FingerprintResult(fp, fp_raw)` — two tables,
same client index, same feature set:

- **`fp`** — the *normalized* fingerprint: Bayesian-shrunk + centered-log-ratio
  (CLR) composition shares, log-transformed scale features. This is the table
  PCA, UMAP, and clustering should consume — the transforms exist specifically to
  make distance and variance behave correctly for those methods.
- **`fp_raw`** — the *non-normalized* fingerprint: the same features in plain,
  business-legible units — actual trade counts, actual dollar amounts, plain
  percentage shares, actual day counts — with no Bayesian shrinkage, no CLR, and
  no log transform applied. This is the table to hand to a non-technical reader,
  or to use in a report or dashboard, where "62% of trades were in equities" or
  "43 trades" reads far more naturally than a CLR value or `log(44) = 3.78`.

**Naming convention:** a feature's column name is identical in both tables,
*except* where the `fp` name carries a `log_` prefix marking a log1p transform
(e.g. `intensity:log_n_trades`) — there, `fp_raw` drops the prefix (`intensity:n_trades`)
since the value is no longer log-transformed. Wherever a feature is already
scale-free (e.g. `direction:buy_share_dollar`, a ratio in [0, 1]), `fp_raw` holds
the exact same value as `fp` — there is nothing to un-transform, so no artificial
distinction is invented. Each feature section below states explicitly, under
"Business-friendly notes", whether its `fp_raw` value differs and how.

Each feature header below is followed by the technical name (or naming pattern)
that column takes in the `fp` fingerprint DataFrame.

## Contents

- [Composition Features](#sec-composition)
  - [Category share profile (count-weighted)](#feat-composition-share)
  - [Attention–money gap](#feat-composition-gap)
- [Directionality](#sec-directionality)
  - [Buy share of dollar volume](#feat-direction-buy-dollar)
  - [Buy share of trade count](#feat-direction-buy-count)
  - [Net-over-gross turnover](#feat-direction-net-over-gross)
- [Concentration / Diversification](#sec-concentration)
  - [HHI (Herfindahl-Hirschman Index)](#feat-concentration-hhi)
  - [Normalized entropy](#feat-concentration-entropy)
  - [Distinct category count](#feat-concentration-n-distinct)
- [Complexity / Sophistication](#sec-complexity)
  - [Derivative share (dollar and trade-count weighted)](#feat-complexity-derivative-share)
  - [Option share of trades](#feat-complexity-option-share)
  - [Uses options (yes/no)](#feat-complexity-uses-options)
- [Intensity (Scale of Activity)](#sec-intensity)
  - [Number of trades](#feat-intensity-n-trades)
  - [Average trade size](#feat-intensity-avg-trade-size)
  - [Total volume traded](#feat-intensity-total-volume)
  - [Trading rate (trades per day)](#feat-intensity-trades-per-day)
  - [Turnover ratio relative to assets under management (optional)](#feat-intensity-turnover-ratio)
- [Temporal Dynamics](#sec-temporal)
  - [Average time between trades](#feat-temporal-mean-gap)
  - [Variability of time between trades](#feat-temporal-cv-gap)
- [How it all comes together](#sec-summary)

---

<a id="sec-composition"></a>
## Composition Features

*(computed once per categorical dimension: asset class, product class, sector)*

Column names below use `{dimension}` (e.g. `asset_class`) and `{category}` (e.g.
`equity`) as placeholders — there is one actual column per category per
dimension, e.g. `asset_class:count:equity`, `asset_class:count:bond`, …

This block answers a simple business question — "across the categories a client
could trade in, where does their activity actually land?" — and does it two ways:
counted by number of trades ("where attention goes") and counted by dollar
amount ("where money goes").

<a id="feat-composition-share"></a>
### Category share profile (count-weighted) (`{dimension}:count:{category}`)

**What it captures:** the client's typical mix of trades across categories — e.g.
a client who does most of their trades in equities versus one who is mostly in
bonds. Clients with very few trades have this estimate pulled gently toward the
overall client population's average mix, so a handful of trades don't get read
as a strong personal preference before there's enough evidence.

**Technical notes:** raw per-category trade-count shares are Bayesian-shrunk
toward the population mean (a Dirichlet/pseudocount prior, strength controlled by
`alpha`), which also guarantees no category share is ever exactly zero. The
shrunk shares are then passed through a centered log-ratio (CLR) transform, which
is what makes ordinary Euclidean distance and PCA behave correctly on this kind
of "parts of a whole" (compositional) data — see `docs/trading-fingerprint-shrinkage-clr.md`
for the full derivation.

**Business-friendly notes (`fp_raw`, same column name):** the plain per-category
share — trade count in that category divided by the client's total trade count
(or dollar volume in that category divided by total dollar volume, for the
dollar-weighted `{dimension}:dollar:{category}` column) — with **no** Bayesian
shrinkage and **no** CLR applied. A value of 0.62 reads directly as "62% of this
client's trades were in equities."

<a id="feat-composition-gap"></a>
### Attention–money gap (`{dimension}:gap:{category}`)

**What it captures:** whether a client's *money* goes to different places than
their *attention* does — for example, someone who places many small trades in
stocks but occasionally makes one very large trade in structured products.
A positive gap for a category means dollars over-weight it relative to trade
count (a few big tickets there); a negative gap means the opposite (many small
tickets, little money).

**Technical notes:** computed as `CLR(dollar shares) − CLR(count shares)`, where
each of the two share vectors is shrunk toward its *own* population mean before
the subtraction. Only two of the three possible blocks — count shares, dollar
shares, and the gap — are ever kept together (default: count + gap), because any
two exactly determine the third; keeping all three would double-count the same
information in any distance calculation or PCA.

**Business-friendly notes (`fp_raw`, same column name):** the plain dollar
share minus the plain count share (both computed as above, no shrinkage, no
CLR) — a straightforward percentage-point gap, e.g. +0.15 reads as "dollars
overweight this category by 15 percentage points relative to trade count."
Unlike the `fp` gap, this is not itself a log-ratio, but the sign and rough
magnitude still tell the same story.

---

<a id="sec-directionality"></a>
## Directionality

Answers: is this client building up positions, selling out of them, or trading
back and forth?

<a id="feat-direction-buy-dollar"></a>
### Buy share of dollar volume (`direction:buy_share_dollar`)

**What it captures:** what fraction of the client's traded dollars were buys
rather than sells. Close to 100% marks an accumulator; close to 0% marks someone
liquidating a position; around 50% marks a two-way trader.

**Technical notes:** `buy dollar volume / total dollar volume`, per client.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — already a
plain, business-legible ratio.

<a id="feat-direction-buy-count"></a>
### Buy share of trade count (`direction:buy_share_count`)

**What it captures:** the same buy-vs-sell question, but counted by number of
trades rather than dollars — useful for spotting clients whose behavior looks
different depending on whether you count tickets or money (e.g. many small sells
offset by one large buy).

**Technical notes:** `buy trade count / total trade count`.

**Business-friendly notes:** same value in both `fp` and `fp_raw`.

<a id="feat-direction-net-over-gross"></a>
### Net-over-gross turnover (`direction:net_over_gross`)

**What it captures:** whether buying and selling roughly cancel out (a churner,
net near zero) or nearly all volume flows one direction (an accumulator or
liquidator, net near +1 or −1).

**Technical notes:** `(buy dollars − sell dollars) / total dollars`, a ratio
bounded in [-1, 1]. Being a ratio already, it needs no additional log-transform
or normalization to be comparable across clients of different sizes.

**Business-friendly notes:** same value in both `fp` and `fp_raw`.

---

<a id="sec-concentration"></a>
## Concentration / Diversification

*(computed once per categorical dimension, both by trade count and by dollar
volume)*

Answers: does this client focus their trading narrowly, or spread it across many
categories?

<a id="feat-concentration-hhi"></a>
### HHI (Herfindahl-Hirschman Index) (`concentration:{dimension}:hhi_count`, `concentration:{dimension}:hhi_dollar`)

**What it captures:** how concentrated a client's activity is. A client putting
almost everything into one category scores high; a client spread evenly across
many categories scores low. Business analogy: the same index used to measure
market concentration, applied here to a single client's own trading mix.

**Technical notes:** sum of squared category shares, computed separately on
count-weighted and dollar-weighted shares.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — HHI is
already computed on plain, unshrunk shares (no Bayesian prior, no CLR).

<a id="feat-concentration-entropy"></a>
### Normalized entropy (`concentration:{dimension}:entropy_count`, `concentration:{dimension}:entropy_dollar`)

**What it captures:** another view of the same concentration-vs-diversification
question, scaled to a clean 0–1 range: 0 means all activity sits in one category,
1 means activity is spread perfectly evenly across every available category.

**Technical notes:** Shannon entropy of the category shares, normalized by
`log(number of categories)`. HHI and entropy tend to move together but weight
concentration differently (HHI penalizes it more sharply); both are kept so
downstream analysis (PCA) can pick up whichever shape best separates clients.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — its
`log(K)` scaling is a self-contained 0–1 normalization, not the Bayesian
shrinkage / CLR pipeline, and the result is already a readable diversification
score.

<a id="feat-concentration-n-distinct"></a>
### Distinct category count (`concentration:{dimension}:n_distinct`)

**What it captures:** simply, how many different categories a client has ever
traded in at all — regardless of how much volume went to each.

**Technical notes:** count of categories with a nonzero trade count for that
client.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — already a
raw count.

---

<a id="sec-complexity"></a>
## Complexity / Sophistication

Answers: does this client stick to plain, simple products, or do they also use
more complex instruments like options and structured products?

<a id="feat-complexity-derivative-share"></a>
### Derivative share (dollar and trade-count weighted) (`complexity:derivative_share_count`, `complexity:derivative_share_dollar`)

**What it captures:** what portion of a client's activity is in derivative or
structured products rather than plain stocks/bonds, used as a rough proxy for
trading sophistication. Both weightings are kept because they can tell very
different stories: a client who dabbles in options on just a few tickets but
keeps almost all their money in plain equity looks quite different from one
running large option positions.

**Technical notes:** share of trade count / dollar volume where the product
class falls in the derivative set (options, structured products).

**Business-friendly notes:** same value in both `fp` and `fp_raw`.

<a id="feat-complexity-option-share"></a>
### Option share of trades (`complexity:option_share_count`)

**What it captures:** narrows the derivative question specifically to options,
since option usage can be a distinct signal from structured-product usage.

**Technical notes:** option trade count / total trade count.

**Business-friendly notes:** same value in both `fp` and `fp_raw`.

<a id="feat-complexity-uses-options"></a>
### Uses options (yes/no) (`complexity:uses_options`)

**What it captures:** a clean flag for whether a client has ever touched an
option at all, regardless of how small that activity is — useful for cleanly
separating "never touches options" clients from everyone else even when their
option share is tiny.

**Technical notes:** binary indicator, 1 if option trade count > 0 else 0.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — already a
plain yes/no flag.

---

<a id="sec-intensity"></a>
## Intensity (Scale of Activity)

Unlike every other block, these features describe raw scale — how big and how
active a client is — rather than trading *style*. They're deliberately kept to a
small handful of features so that sheer size doesn't drown out the more
behavioral signals from the other blocks.

<a id="feat-intensity-n-trades"></a>
### Number of trades (`intensity:log_n_trades`)

**What it captures:** how many trades a client has made in total — a basic
measure of overall activity level.

**Technical notes:** log(1 + trade count). The log transform tames the heavy
right skew typical of trade counts (a few very active clients would otherwise
dominate purely on magnitude).

**Business-friendly notes (`fp_raw` column `intensity:n_trades`):** the actual
trade count, no log transform — "43 trades" rather than `log(44) = 3.78`.

<a id="feat-intensity-avg-trade-size"></a>
### Average trade size (`intensity:log_avg_trade_size`)

**What it captures:** the typical size of a client's individual trades — 
distinguishing a client who places a few large trades from one who places many
small ones, independent of how many trades they make in total.

**Technical notes:** log(1 + total dollar volume / trade count).

**Business-friendly notes (`fp_raw` column `intensity:avg_trade_size`):** the
actual average trade size in dollars, no log transform.

<a id="feat-intensity-total-volume"></a>
### Total volume traded (`intensity:log_total_volume`)

**What it captures:** the overall dollar footprint of a client's trading
activity.

**Technical notes:** log(1 + total dollar volume summed across all trades).

**Business-friendly notes (`fp_raw` column `intensity:total_volume`):** the
actual total dollar volume, no log transform.

<a id="feat-intensity-trades-per-day"></a>
### Trading rate (trades per day) (`intensity:log_trades_per_day`)

**What it captures:** how frequently a client trades, adjusted for how long
they've actually been observed — so a client who made 50 trades in one week
reads differently from one who made 50 trades over a year.

**Technical notes:** log(1 + trade count / observed span in days), where the
observed span is floored at one day to avoid dividing by zero for clients seen
only briefly.

**Business-friendly notes (`fp_raw` column `intensity:trades_per_day`):** the
actual trades-per-day rate, no log transform.

<a id="feat-intensity-turnover-ratio"></a>
### Turnover ratio relative to assets under management (optional) (`intensity:log_turnover_ratio`)

**What it captures:** trading volume scaled by the size of the client's overall
portfolio, so a small account that trades its entire book looks comparable to a
large account doing the same thing, rather than the large account automatically
"winning" on raw dollar volume.

**Technical notes:** log(1 + total dollar volume / AUM); only computed when an
AUM figure per client is supplied to the pipeline.

**Business-friendly notes (`fp_raw` column `intensity:turnover_ratio`):** the
actual volume/AUM ratio, no log transform; also only present when AUM is
supplied.

---

<a id="sec-temporal"></a>
## Temporal Dynamics

Answers: does this client trade on a steady rhythm, or in irregular bursts?

<a id="feat-temporal-mean-gap"></a>
### Average time between trades (`temporal:log_mean_gap_days`)

**What it captures:** on average, how many days pass between one trade and the
next for this client — a general read on trading cadence.

**Technical notes:** log(1 + mean inter-trade gap in days). Log-transformed
since the raw gap is a time scale that correlates with overall activity level.

**Business-friendly notes (`fp_raw` column `temporal:mean_gap_days`):** the
actual mean gap in days, no log transform — "a trade every 12 days" rather
than `log(13) = 2.56`.

<a id="feat-temporal-cv-gap"></a>
### Variability of time between trades (`temporal:cv_gap`)

**What it captures:** distinguishes clients who trade on a steady, predictable
schedule (e.g. periodic rebalancing or coupon reinvestment) from "bursty"
traders who go quiet for long stretches and then trade in clusters (e.g. reacting
to news or market dips).

**Technical notes:** coefficient of variation (standard deviation / mean) of
inter-trade gaps — a scale-free ratio needing no further transform. Values much
less than 1 indicate steady/metronomic timing; values at or above 1 indicate
bursty timing. Clients with fewer than two trades have no defined gap and are
median-imputed when the full fingerprint is assembled.

**Business-friendly notes:** same value in both `fp` and `fp_raw` — already
scale-free, nothing to un-transform. Median-imputed independently in each
table for clients with fewer than two trades.

---

<a id="sec-summary"></a>
## How it all comes together

All six blocks are computed independently per client and then joined side by
side into one feature table, keyed on client ID — twice over, once for `fp` and
once for `fp_raw`, since every block builder returns a `(normalized, raw)` pair.
`build_fingerprint(...)` assembles both into a `FingerprintResult(fp, fp_raw)`
(usable either as `fp, fp_raw = build_fingerprint(...)` or
`result.fp` / `result.fp_raw`). Any values that can't be computed for a given
client (for example, temporal variability for a client with only one trade) are
filled in with the median value across all clients — independently within each
table — so the downstream analysis (PCA, UMAP, clustering, which consume `fp`)
and any business-facing reporting (which should consume `fp_raw`) both always
see a complete table.

See `docs/trading-fingerprint-shrinkage-clr.md` for the full mathematical
treatment of the shrinkage and CLR steps used in the composition block.
