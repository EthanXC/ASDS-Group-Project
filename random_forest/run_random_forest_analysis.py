"""Random-forest analysis for the Hillstrom email experiment.

Research question: can pre-campaign customer characteristics identify who benefits
more from the Men's or the Women's email, and does assigning emails by a learned rule
raise visits out of sample compared with sending everyone the single best email?

Design
  1. Keep the two email arms (random assignment, propensity 1/2) and split them 50/50
     into a training half and a held-out test half. The test half is used once.
  2. On the training half, choose between two random-forest CATE learners and their
     leaf size by 5-fold cross-validated transformed-outcome risk.
  3. Learned rule: send the Men's email when tau_hat(x) > 0, else the Women's email.
     An interpretable depth-2 tree is also distilled from the forest's predictions.
  4. On the test half, estimate each rule's visit rate with AIPW scores and compare it
     with "everyone gets the email that did best on the training half" (paired CI).
  5. Repeat steps 1, 3 and 4 over many random splits to show split-to-split variability.

Run from the repository root:  python random_forest/run_random_forest_analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

from hillstrom_data import (
    MENS_EMAIL_LABEL, NO_EMAIL_LABEL, WOMENS_EMAIL_LABEL,
    build_covariate_matrix, email_recipients_only, load_hillstrom,
)
from uplift_forest import (
    LEARNER_FITTERS, aipw_scores, cate_risk, cross_fitted_cate,
    estimate_policy_value, estimate_policy_value_difference,
    fit_two_forest_learner, group_average_treatment_effects,
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
PRIMARY_SPLIT_SEED = 2026
NUMBER_OF_REPEATED_SPLITS = 20
LEAF_SIZE_GRID = [100, 200, 400, 800, 1600, 3200]
TREE_DEPTH = 2
TREE_MIN_LEAF_SHARE = 0.05  # each interpretable segment holds >= 5% of customers
Z_95 = stats.norm.ppf(0.975)

# Same email colors as the rest of the team's slides, plus neutral inks.
MENS_EMAIL_COLOR = "#1b6ca8"
WOMENS_EMAIL_COLOR = "#d1495b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
NEUTRAL_MARK_COLOR = "#8a8984"


def split_train_test(number_of_customers, received_mens_email, visited, split_seed):
    all_rows = np.arange(number_of_customers)
    return train_test_split(
        all_rows, test_size=0.5, stratify=2 * received_mens_email + visited,
        random_state=split_seed,
    )


def tune_learners_by_cross_validation(covariates, received_mens_email, visited, random_seed):
    """CV transformed-outcome risk for every (learner, leaf size) pair, reported
    relative to the constant-effect benchmark tau_hat(x) = training ATE (negative =
    better than assuming everyone responds the same)."""
    average_effect = visited[received_mens_email == 1].mean() - visited[received_mens_email == 0].mean()
    constant_effect_risk = cate_risk(received_mens_email, visited, np.full(len(visited), average_effect))
    rows = []
    for learner_name in LEARNER_FITTERS:
        for min_samples_leaf in LEAF_SIZE_GRID:
            out_of_fold_cate = cross_fitted_cate(
                learner_name, covariates, received_mens_email, visited, min_samples_leaf, random_seed
            )
            rows.append({
                "learner": learner_name,
                "min_samples_leaf": min_samples_leaf,
                "risk_minus_constant_effect_risk": cate_risk(received_mens_email, visited, out_of_fold_cate)
                - constant_effect_risk,
                "share_assigned_womens_email": float(np.mean(out_of_fold_cate <= 0)),
            })
    return pd.DataFrame(rows)


def evaluate_one_split(covariates, received_mens_email, visited, split_seed,
                       chosen_learner, chosen_leaf, nuisance_leaf):
    """Train on one half, evaluate assignment rules on the other half."""
    training_rows, test_rows = split_train_test(len(visited), received_mens_email, visited, split_seed)
    covariates_train, covariates_test = covariates.iloc[training_rows], covariates.iloc[test_rows]
    treatment_train, treatment_test = received_mens_email[training_rows], received_mens_email[test_rows]
    visited_train, visited_test = visited[training_rows], visited[test_rows]

    cate_learner = LEARNER_FITTERS[chosen_learner](
        covariates_train, treatment_train, visited_train, chosen_leaf, split_seed
    )
    test_cate = cate_learner.predict_cate(covariates_test)

    # Interpretable rule: shallow regression tree fit to cross-fitted tau_hat on the
    # training half (cross-fitting keeps the tree from copying in-sample noise).
    training_cate = cross_fitted_cate(
        chosen_learner, covariates_train, treatment_train, visited_train, chosen_leaf, split_seed
    )
    interpretable_tree = DecisionTreeRegressor(
        max_depth=TREE_DEPTH,
        min_samples_leaf=int(TREE_MIN_LEAF_SHARE * len(training_rows)),
        random_state=split_seed,
    ).fit(covariates_train, training_cate)

    # Outcome models for the AIPW scores, fit on the training half only.
    nuisance_model = fit_two_forest_learner(
        covariates_train, treatment_train, visited_train, nuisance_leaf, split_seed + 1000
    )
    mu_mens_test, mu_womens_test = nuisance_model.predict_visit_probabilities(covariates_test)
    gamma_mens, gamma_womens = aipw_scores(mu_mens_test, mu_womens_test, treatment_test, visited_test)

    # The "single best email" is chosen on the training half, not peeked at on test.
    mens_email_won_training = visited_train[treatment_train == 1].mean() > visited_train[treatment_train == 0].mean()
    number_of_test_customers = len(test_rows)
    assignment_rules = {
        "Everyone: Men's email": np.ones(number_of_test_customers, dtype=bool),
        "Everyone: Women's email": np.zeros(number_of_test_customers, dtype=bool),
        "Forest rule": test_cate > 0,
        "Depth-2 tree rule": interpretable_tree.predict(covariates_test) > 0,
    }
    best_single_email_rule = (
        "Everyone: Men's email" if mens_email_won_training else "Everyone: Women's email"
    )

    policy_rows = []
    for rule_name, assign_mens_email in assignment_rules.items():
        value, value_se = estimate_policy_value(assign_mens_email, gamma_mens, gamma_womens)
        gain, gain_se = estimate_policy_value_difference(
            assign_mens_email, assignment_rules[best_single_email_rule], gamma_mens, gamma_womens
        )
        policy_rows.append({
            "rule": rule_name,
            "share_sent_womens_email": float(1 - assign_mens_email.mean()),
            "visit_rate": value,
            "visit_rate_se": value_se,
            "gain_vs_best_single_email": gain,
            "gain_se": gain_se,
            "one_sided_p_value_gain_positive": float(stats.norm.sf(gain / gain_se)) if gain_se > 0 else np.nan,
        })

    return {
        "policy_table": pd.DataFrame(policy_rows),
        "best_single_email_rule": best_single_email_rule,
        "test_cate": test_cate,
        "gamma_mens": gamma_mens,
        "gamma_womens": gamma_womens,
        "covariates_test": covariates_test,
        "interpretable_tree": interpretable_tree,
        "covariate_names": list(covariates.columns),
        "number_of_test_customers": number_of_test_customers,
    }


# ----------------------------------------------------------------------------- plots

def style_axes(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(NEUTRAL_MARK_COLOR)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axes.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.title.set_color(TEXT_PRIMARY)


def plot_cate_distribution(test_cate, output_path):
    effect_in_points = 100 * test_cate
    # Bins aligned so that 0 is an edge: every bar is wholly on one side of the rule.
    bin_width = (effect_in_points.max() - effect_in_points.min()) / 40
    bin_edges = bin_width * np.arange(
        np.floor(effect_in_points.min() / bin_width), np.ceil(effect_in_points.max() / bin_width) + 1
    )
    figure, axes = plt.subplots(figsize=(6.5, 3.6))
    counts, edges, bars = axes.hist(effect_in_points, bins=bin_edges, edgecolor="white", linewidth=1)
    for bar, left_edge in zip(bars, edges[:-1]):
        bar.set_facecolor(MENS_EMAIL_COLOR if left_edge >= 0 else WOMENS_EMAIL_COLOR)
    axes.set_ylim(0, counts.max() * 1.18)  # headroom for the two labels
    axes.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    share_womens = np.mean(test_cate <= 0)
    axes.text(0.02, 0.95, f"Women's email better: {share_womens:.0%}", transform=axes.transAxes,
              color=TEXT_PRIMARY, fontsize=9, va="top")
    axes.text(0.98, 0.95, f"Men's email better: {1 - share_womens:.0%}", transform=axes.transAxes,
              color=TEXT_PRIMARY, fontsize=9, va="top", ha="right")
    axes.set(xlabel="Predicted visit lift of Men's over Women's email, τ̂(x) (percentage points)",
             ylabel="Test customers")
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_group_average_effects(group_effects, overall_effect, output_path):
    figure, axes = plt.subplots(figsize=(6.5, 3.6))
    groups = group_effects["group"]
    axes.errorbar(groups, 100 * group_effects["estimated_effect"],
                  yerr=100 * Z_95 * group_effects["standard_error"], fmt="o", color=MENS_EMAIL_COLOR,
                  markersize=8, capsize=4, linewidth=2, label="Estimated on test half (AIPW, 95% CI)")
    axes.plot(groups, 100 * group_effects["mean_predicted_effect"], "D", markersize=7,
              markerfacecolor="white", markeredgecolor=TEXT_SECONDARY, markeredgewidth=1.5,
              label="Forest prediction τ̂(x), group mean")
    axes.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    axes.axhline(100 * overall_effect, color=NEUTRAL_MARK_COLOR, linewidth=1, linestyle="--",
                 label="Average effect, all test customers")
    axes.set_xticks(groups, [f"Q{g}" for g in groups])
    axes.set(xlabel="Test customers grouped by predicted effect (Q1 = most Women's-leaning)",
             ylabel="Men's − Women's visit rate (pp)")
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_policy_values(policy_table, output_path):
    figure, axes = plt.subplots(figsize=(6.5, 3.0))
    ordered = policy_table.iloc[::-1].reset_index(drop=True)
    colors = [WOMENS_EMAIL_COLOR if "Women" in rule else MENS_EMAIL_COLOR if "Men" in rule
              else TEXT_PRIMARY for rule in ordered["rule"]]
    for row_index, row in ordered.iterrows():
        axes.errorbar(100 * row["visit_rate"], row_index, xerr=100 * Z_95 * row["visit_rate_se"],
                      fmt="o", color=colors[row_index], markersize=8, capsize=4, linewidth=2)
        axes.text(100 * (row["visit_rate"] + Z_95 * row["visit_rate_se"]) + 0.15, row_index,
                  f"{100 * row['visit_rate']:.2f}%", va="center", fontsize=9, color=TEXT_PRIMARY)
    axes.set_yticks(range(len(ordered)), ordered["rule"])
    axes.set(xlabel="Estimated two-week visit rate on the test half (AIPW, 95% CI)")
    style_axes(axes)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_repeated_split_gains(repeated_gains, output_path):
    figure, axes = plt.subplots(figsize=(6.5, 3.4))
    ordered = repeated_gains.sort_values("gain_vs_best_single_email").reset_index(drop=True)
    ci_excludes_zero = (ordered["gain_vs_best_single_email"] - Z_95 * ordered["gain_se"]) > 0
    axes.errorbar(ordered.index + 1, 100 * ordered["gain_vs_best_single_email"],
                  yerr=100 * Z_95 * ordered["gain_se"], fmt="none", ecolor=NEUTRAL_MARK_COLOR,
                  capsize=3, linewidth=1.5)
    axes.scatter(ordered.index + 1, 100 * ordered["gain_vs_best_single_email"], s=40, zorder=3,
                 c=[MENS_EMAIL_COLOR if excludes else "white" for excludes in ci_excludes_zero],
                 edgecolors=MENS_EMAIL_COLOR, linewidths=1.5)
    axes.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    axes.set_xticks(ordered.index + 1)
    axes.set(xlabel=f"Random train/test split (sorted; filled = 95% CI above 0, "
                    f"{int(ci_excludes_zero.sum())} of {len(ordered)})",
             ylabel="Forest rule − best single email (pp)")
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# ----------------------------------------------------------------------------- report

def format_percent(value, digits=2):
    return f"{100 * value:.{digits}f}%"


def format_points(value, standard_error):
    low, high = value - Z_95 * standard_error, value + Z_95 * standard_error
    return f"{100 * value:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]"


def describe_segments_by_cate(test_cate, covariates_test):
    """Mean predicted effect within simple, interpretable subgroups of test customers."""
    segments = {
        "Bought only women's merchandise": (covariates_test["womens"] == 1) & (covariates_test["mens"] == 0),
        "Bought only men's merchandise": (covariates_test["mens"] == 1) & (covariates_test["womens"] == 0),
        "Bought both": (covariates_test["mens"] == 1) & (covariates_test["womens"] == 1),
        "New customer (newbie = 1)": covariates_test["newbie"] == 1,
        "Returning customer (newbie = 0)": covariates_test["newbie"] == 0,
    }
    rows = []
    for segment_name, in_segment in segments.items():
        rows.append({
            "segment": segment_name,
            "share_of_customers": float(in_segment.mean()),
            "mean_predicted_effect": float(test_cate[in_segment.to_numpy()].mean()),
            "share_sent_womens_email_by_forest": float(np.mean(test_cate[in_segment.to_numpy()] <= 0)),
        })
    return pd.DataFrame(rows)


def write_report(customers, tuning_table, chosen, primary, group_effects, overall_effect,
                 segment_table, repeated_gains, output_path):
    arm_summary = customers.groupby("segment")["visit"].agg(["mean", "count"])
    lines = [
        "# Random forest results: who benefits from the Men's vs Women's email?",
        "",
        "Generated by `random_forest/run_random_forest_analysis.py`; all numbers are reproducible.",
        "",
        "## Experiment context (all 64,000 customers)",
        "",
        "| Arm | Customers | Two-week visit rate |",
        "|---|---|---|",
    ]
    for arm in [MENS_EMAIL_LABEL, WOMENS_EMAIL_LABEL, NO_EMAIL_LABEL]:
        lines.append(f"| {arm} | {int(arm_summary.loc[arm, 'count']):,} | {format_percent(arm_summary.loc[arm, 'mean'])} |")
    lines += [
        "",
        "## Model selection on the training half (5-fold CV)",
        "",
        "Risk is the transformed-outcome loss minus that of a constant-effect model "
        "(τ̂(x) = training ATE). Negative means the forest beats \"everyone responds the same\".",
        "",
        "| Learner | min_samples_leaf | Risk − constant-effect risk | Share sent Women's |",
        "|---|---|---|---|",
    ]
    for _, row in tuning_table.iterrows():
        lines.append(f"| {row['learner']} | {row['min_samples_leaf']} | "
                     f"{row['risk_minus_constant_effect_risk']:+.5f} | {row['share_assigned_womens_email']:.0%} |")
    lines += [
        "",
        f"**Chosen:** `{chosen['learner']}` with min_samples_leaf = {chosen['leaf']}. "
        f"AIPW outcome models: two-forest learner with min_samples_leaf = {chosen['nuisance_leaf']}.",
        "",
        f"## Held-out test half (n = {primary['number_of_test_customers']:,}, split seed {PRIMARY_SPLIT_SEED})",
        "",
        f"Best single email on the training half: **{primary['best_single_email_rule']}**.",
        "",
        "| Rule | Share sent Women's | Visit rate (95% CI) | Gain vs best single email (95% CI) | One-sided p |",
        "|---|---|---|---|---|",
    ]
    for _, row in primary["policy_table"].iterrows():
        visit_ci = (f"{format_percent(row['visit_rate'])} "
                    f"[{format_percent(row['visit_rate'] - Z_95 * row['visit_rate_se'])}, "
                    f"{format_percent(row['visit_rate'] + Z_95 * row['visit_rate_se'])}]")
        p_value = "—" if np.isnan(row["one_sided_p_value_gain_positive"]) else f"{row['one_sided_p_value_gain_positive']:.3f}"
        lines.append(f"| {row['rule']} | {row['share_sent_womens_email']:.0%} | {visit_ci} | "
                     f"{format_points(row['gain_vs_best_single_email'], row['gain_se'])} | {p_value} |")
    lines += [
        "",
        "### Depth-2 tree distilled from the forest (leaf values are predicted τ̂)",
        "",
        "```",
        export_text(primary["interpretable_tree"], feature_names=primary["covariate_names"], decimals=4).rstrip(),
        "```",
        "",
        "### Group average effects on the test half (quintiles of τ̂)",
        "",
        f"Average Men's − Women's effect, all test customers: {100 * overall_effect:+.2f} pp.",
        "",
        "| Quintile | Customers | Mean predicted τ̂ | Estimated effect (95% CI) |",
        "|---|---|---|---|",
    ]
    for _, row in group_effects.iterrows():
        lines.append(f"| Q{int(row['group'])} | {int(row['customers']):,} | {100 * row['mean_predicted_effect']:+.2f} pp | "
                     f"{format_points(row['estimated_effect'], row['standard_error'])} |")
    top_minus_bottom = group_effects.iloc[-1]["estimated_effect"] - group_effects.iloc[0]["estimated_effect"]
    top_minus_bottom_se = np.hypot(group_effects.iloc[-1]["standard_error"], group_effects.iloc[0]["standard_error"])
    lines += [
        "",
        f"Top − bottom quintile: {format_points(top_minus_bottom, top_minus_bottom_se)}, "
        f"two-sided p = {2 * stats.norm.sf(abs(top_minus_bottom / top_minus_bottom_se)):.4f}.",
        "",
        "### Predicted effect by subgroup (test half)",
        "",
        "| Subgroup | Share of customers | Mean τ̂ | Share forest sends Women's |",
        "|---|---|---|---|",
    ]
    for _, row in segment_table.iterrows():
        lines.append(f"| {row['segment']} | {row['share_of_customers']:.0%} | {100 * row['mean_predicted_effect']:+.2f} pp | "
                     f"{row['share_sent_womens_email_by_forest']:.0%} |")
    gains = repeated_gains["gain_vs_best_single_email"]
    significant_splits = int(((gains - Z_95 * repeated_gains["gain_se"]) > 0).sum())
    lines += [
        "",
        f"## Stability across {len(repeated_gains)} random 50/50 splits (forest rule vs best single email)",
        "",
        f"Mean gain {100 * gains.mean():+.2f} pp; median {100 * gains.median():+.2f} pp; "
        f"range [{100 * gains.min():+.2f}, {100 * gains.max():+.2f}] pp; positive in "
        f"{int((gains > 0).sum())} of {len(gains)} splits; 95% CI above zero in {significant_splits} of {len(gains)}.",
        "",
        "Hyperparameters were tuned once on the primary training half and held fixed across "
        "splits, so the other splits' test halves overlap the data used for tuning (a mild "
        "optimistic bias).",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    customers = load_hillstrom()
    email_recipients = email_recipients_only(customers)
    covariates = build_covariate_matrix(email_recipients)
    received_mens_email = email_recipients["received_mens_email"].to_numpy()
    visited = email_recipients["visit"].to_numpy()

    training_rows, _ = split_train_test(len(visited), received_mens_email, visited, PRIMARY_SPLIT_SEED)
    print("Tuning random forests by cross-validation on the training half ...")
    tuning_table = tune_learners_by_cross_validation(
        covariates.iloc[training_rows], received_mens_email[training_rows], visited[training_rows],
        PRIMARY_SPLIT_SEED,
    )
    print(tuning_table.to_string(index=False))
    best_overall = tuning_table.loc[tuning_table["risk_minus_constant_effect_risk"].idxmin()]
    two_forest_rows = tuning_table[tuning_table["learner"] == "two_forest"]
    chosen = {
        "learner": best_overall["learner"],
        "leaf": int(best_overall["min_samples_leaf"]),
        "nuisance_leaf": int(two_forest_rows.loc[two_forest_rows["risk_minus_constant_effect_risk"].idxmin(), "min_samples_leaf"]),
    }
    print(f"Chosen: {chosen}")

    primary = evaluate_one_split(covariates, received_mens_email, visited, PRIMARY_SPLIT_SEED,
                                 chosen["learner"], chosen["leaf"], chosen["nuisance_leaf"])
    print(primary["policy_table"].to_string(index=False))

    group_effects = group_average_treatment_effects(primary["test_cate"], primary["gamma_mens"], primary["gamma_womens"])
    overall_effect = float(np.mean(primary["gamma_mens"] - primary["gamma_womens"]))
    segment_table = describe_segments_by_cate(primary["test_cate"], primary["covariates_test"])

    print(f"Repeating over {NUMBER_OF_REPEATED_SPLITS} random splits ...")
    repeated_rows = []
    for repeat_index in range(NUMBER_OF_REPEATED_SPLITS):
        split_result = evaluate_one_split(covariates, received_mens_email, visited, 10_000 + repeat_index,
                                          chosen["learner"], chosen["leaf"], chosen["nuisance_leaf"])
        forest_row = split_result["policy_table"].set_index("rule").loc["Forest rule"]
        repeated_rows.append({"split_seed": 10_000 + repeat_index,
                              "gain_vs_best_single_email": forest_row["gain_vs_best_single_email"],
                              "gain_se": forest_row["gain_se"]})
    repeated_gains = pd.DataFrame(repeated_rows)

    plot_cate_distribution(primary["test_cate"], OUTPUT_DIRECTORY / "cate_distribution.png")
    plot_group_average_effects(group_effects, overall_effect, OUTPUT_DIRECTORY / "group_average_effects.png")
    plot_policy_values(primary["policy_table"], OUTPUT_DIRECTORY / "policy_values.png")
    plot_repeated_split_gains(repeated_gains, OUTPUT_DIRECTORY / "repeated_split_gains.png")
    primary["policy_table"].to_csv(OUTPUT_DIRECTORY / "policy_values.csv", index=False)
    group_effects.to_csv(OUTPUT_DIRECTORY / "group_average_effects.csv", index=False)
    repeated_gains.to_csv(OUTPUT_DIRECTORY / "repeated_split_gains.csv", index=False)
    write_report(customers, tuning_table, chosen, primary, group_effects, overall_effect,
                 segment_table, repeated_gains, OUTPUT_DIRECTORY / "results.md")
    print(f"Wrote figures and results.md to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
