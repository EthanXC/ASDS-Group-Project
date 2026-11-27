# Random forest analysis: Hillstrom email experiment

**Question.** Can pre-campaign customer characteristics identify who benefits more from
the Men's or the Women's email, and does assigning emails by a learned rule raise visits
out of sample compared with sending everyone the single best email?

## How to run

From the repository root (Python 3 with numpy, pandas, scikit-learn, scipy, matplotlib):

```
python random_forest/run_random_forest_analysis.py   # ~4 min; writes random_forest/outputs/
python -m unittest discover -s random_forest -v      # sanity checks, ~3 s
```

The CSV downloads to `data/hillstrom_2008.csv` on first run (same mirror as the course
starter notebook). Every random step is seeded, so results reproduce exactly.

## Files

| File | What it does |
|---|---|
| `hillstrom_data.py` | Load + clean (fixes "Surburban"), keep the two email arms, build covariates |
| `uplift_forest.py` | Forest CATE learners, cross-fitting, CV risk, AIPW policy evaluation, group effects |
| `run_random_forest_analysis.py` | End-to-end pipeline, figures, and `outputs/results.md` |
| `test_uplift_forest.py` | Unit tests incl. simulations with a known treatment effect |

## Method in brief

Notation: $T_i=1$ Men's email, $T_i=0$ Women's email, $Y_i$ = visited within two weeks,
$X_i$ = pre-campaign covariates (recency, history, mens, womens, newbie, zip_code,
channel). Propensity $e = P(T_i=1\mid X_i) = 1/2$ by randomization.

1. **Target.** CATE $\tau(x) = E[Y_i(1) - Y_i(0) \mid X_i = x]$. A rule
   $\pi(x)\in\{0,1\}$ has value $V(\pi) = E[Y_i(\pi(X_i))]$, the visit rate if it were used.
2. **Two random-forest learners.**
   *Two-forest (T-learner):* fit $\hat\mu_1, \hat\mu_0$ on each arm, $\hat\tau = \hat\mu_1 - \hat\mu_0$.
   *Transformed outcome:* $Z_i = Y_i (T_i - e)/(e(1-e))$ satisfies $E[Z_i\mid X_i]=\tau(X_i)$, so
   one forest regressing $Z$ on $X$ estimates $\tau$ directly.
3. **Tuning without seeing $\tau$.** $E[(Z - \hat\tau(X))^2] = E[(\tau(X) - \hat\tau(X))^2] + \text{const}$,
   so 5-fold CV of this loss on the training half picks the learner and `min_samples_leaf`.
4. **Learned rule.** $\hat\pi(x) = 1\{\hat\tau(x) > 0\}$, plus a depth-2 tree fit to
   cross-fitted $\hat\tau$ as an interpretable version.
5. **Honest evaluation.** On the untouched test half, AIPW scores
   $\Gamma_i(t) = \hat\mu_t(X_i) + \frac{1\{T_i=t\}}{P(T_i=t)}(Y_i - \hat\mu_t(X_i))$ give
   $\hat V(\pi) = \frac1n\sum_i \Gamma_i(\pi(X_i))$, unbiased because $e$ is known.
   Gains vs. the best single email are paired differences with CLT standard errors.
6. **Checks.** Group average effects by quintile of $\hat\tau$ (does the ranking hold up on
   new customers?), and 20 repeated random splits.

## Headline result (see `outputs/results.md` for all numbers)

- **Heterogeneity is real.** Test customers in the top quintile of $\hat\tau$ gain about
  +6.7 pp from Men's over Women's; the bottom quintile is about −0.4 pp (top − bottom
  +7.1 pp, 95% CI [+3.9, +10.3]). The split is driven by purchase history: past buyers of
  men's merchandise respond far more to the Men's email.
- **Targeting doesn't pay.** The learned rule sends ~30% of customers (mostly women's-only
  buyers) the Women's email, but for them the two emails are roughly tied. Out of sample the
  rule's visit rate is 18.20% vs 18.27% for everyone getting the Men's email
  (gain −0.07 pp, 95% CI [−0.61, +0.48]); across 20 splits the mean gain is +0.08 pp and no
  split's CI excludes zero.
- **Takeaway.** Large differences in *how much* people respond do not create value from
  targeting unless the *best email flips sign* for a sizable group. Here it barely does.

## Limitations

- Outcome is a two-week visit; conversion and spend are rarer and may rank differently.
- The rule only chooses between the two emails; it never recommends "no email".
- Hyperparameters were tuned on the primary split and reused in the repeated splits.
- Results generalize to this retailer's 2008 customer base and email creative only.
