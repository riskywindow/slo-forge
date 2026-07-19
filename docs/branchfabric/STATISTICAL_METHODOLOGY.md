# BranchFabric Characterization Statistical Methodology

This methodology separates measurement evidence from controlled synthetic
projections. Every sample series records its experiment, raw artifact path and
SHA-256, evidence class, selector, explicit seed, and repetition. The ordered
warmup and measurement samples remain in the series model. Derived analyses
exclude only samples designated by the recorded warmup policy; they never edit
the raw artifact.

## Repetition and run order

Performance experiments declare warmup before execution, use multiple explicit
seeds and repetitions, and randomize matrix run order with the matrix seed.
Background load, clocks, and machine manifests belong beside the raw samples.
Warmup values are retained even though they are excluded from summaries.

## Descriptive statistics

Reports include the median, p90, p95, p99, maximum, and sample count. Percentiles
use Hyndman-Fan type 7 interpolation. Empirical CDFs are exact when the number of
unique values fits the declared output limit; larger CDFs use a deterministic
rank grid and are marked non-exact. Equal-width histograms reject bounds that
would omit a sample.

## Uncertainty and effects

Confidence intervals use a deterministic percentile bootstrap with an explicit
seed, confidence level, repetition count, and a safety bound on total resampling
draws. Paired comparisons preserve experiment order and report candidate minus
baseline mean and median differences, paired Cohen's dz, matched-pairs rank
biserial effect, and relative change against the absolute baseline magnitude
where the baseline is nonzero. Relative
pairs with zero baselines are counted and reported as undefined rather than
silently coerced.

Pearson and tie-aware Spearman correlations are reported for paired series.
Correlation for a constant series is explicitly `null`.

## Noise and outliers

Noise-floor summaries use all post-warmup samples and report MAD, p95 absolute
deviation, population standard deviation, and relative forms when their
denominators are nonzero. MAD outlier analysis only identifies sample indices
and values. It records that zero samples were removed. A zero-MAD series flags
non-median values without manufacturing infinite scores.

Outliers may be investigated as scheduling, thermal, contention, or fault
signals. They are not deleted from the primary analysis. If a sensitivity view
is later required, it must be an additional result with its own method and
provenance, never a replacement for raw evidence.

## Resource bounds

The standard-library implementation accepts at most one million values per raw
series. ECDF points, histogram bins, bootstrap repetitions, and total bootstrap
draws have separate hard bounds. These bounds prevent analysis configuration
from creating unbounded CPU or memory work while preserving large raw traces in
their bulk trace format.
