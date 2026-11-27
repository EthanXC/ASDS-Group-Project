"""Random-forest estimates of who benefits more from the Men's vs Women's email, and
out-of-sample evaluation of email-assignment rules.

Notation (potential outcomes, restricted to email recipients):
    X_i            pre-campaign covariates of customer i
    T_i in {0, 1}  1 if customer i was sent the Men's email, 0 if the Women's email
    Y_i in {0, 1}  1 if customer i visited the website within two weeks
    Y_i(1), Y_i(0) potential visit outcomes under the Men's / Women's email
    e = P(T_i = 1 | X_i) = 1/2   known by design (random assignment)
    mu_t(x) = E[Y_i(t) | X_i = x]         arm-specific visit probability
    tau(x)  = mu_1(x) - mu_0(x)           conditional average treatment effect (CATE):
                                          how much more likely to visit under Men's
    pi(x) in {0, 1}                       an assignment rule (1 = send Men's email)
    V(pi) = E[Y_i(pi(X_i))]               policy value: visit rate if rule pi is used
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

PROPENSITY_MENS_EMAIL = 0.5


def make_random_forest(min_samples_leaf: int, random_seed: int, n_estimators: int = 300):
    """Regression forest; on a 0/1 outcome its leaf means are probability estimates.
    `min_samples_leaf` is the main complexity knob and is tuned by cross-validation."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=0.5,
        n_jobs=-1,
        random_state=random_seed,
    )


@dataclass
class TwoForestLearner:
    """T-learner: one forest per arm, tau_hat(x) = mu_hat_1(x) - mu_hat_0(x)."""

    mens_email_forest: RandomForestRegressor
    womens_email_forest: RandomForestRegressor

    def predict_visit_probabilities(self, covariates):
        return (
            self.mens_email_forest.predict(covariates),
            self.womens_email_forest.predict(covariates),
        )

    def predict_cate(self, covariates):
        mu_mens, mu_womens = self.predict_visit_probabilities(covariates)
        return mu_mens - mu_womens


@dataclass
class TransformedOutcomeLearner:
    """Single forest fit to the transformed outcome Z, whose mean given X is tau(X)."""

    cate_forest: RandomForestRegressor

    def predict_cate(self, covariates):
        return self.cate_forest.predict(covariates)


def fit_two_forest_learner(covariates, received_mens_email, visited, min_samples_leaf, random_seed):
    received_mens_email = np.asarray(received_mens_email)
    visited = np.asarray(visited)
    mens_arm = received_mens_email == 1
    mens_email_forest = make_random_forest(min_samples_leaf, random_seed)
    womens_email_forest = make_random_forest(min_samples_leaf, random_seed + 1)
    mens_email_forest.fit(covariates[mens_arm], visited[mens_arm])
    womens_email_forest.fit(covariates[~mens_arm], visited[~mens_arm])
    return TwoForestLearner(mens_email_forest, womens_email_forest)


def transformed_outcome(received_mens_email, visited, propensity=PROPENSITY_MENS_EMAIL):
    """Z_i = Y_i * (T_i - e) / (e (1 - e)).

    With known propensity e, E[Z_i | X_i = x] = tau(x), so Z is an unbiased (but very
    noisy) per-customer signal of the CATE. With e = 1/2, Z_i is +2Y_i or -2Y_i.
    """
    received_mens_email = np.asarray(received_mens_email, dtype=float)
    visited = np.asarray(visited, dtype=float)
    return visited * (received_mens_email - propensity) / (propensity * (1 - propensity))


def fit_transformed_outcome_learner(covariates, received_mens_email, visited, min_samples_leaf, random_seed):
    cate_forest = make_random_forest(min_samples_leaf, random_seed)
    cate_forest.fit(covariates, transformed_outcome(received_mens_email, visited))
    return TransformedOutcomeLearner(cate_forest)


LEARNER_FITTERS = {
    "two_forest": fit_two_forest_learner,
    "transformed_outcome": fit_transformed_outcome_learner,
}


def cross_fitted_cate(learner_name, covariates, received_mens_email, visited,
                      min_samples_leaf, random_seed, number_of_folds=5):
    """tau_hat for every customer from a model that never saw that customer."""
    received_mens_email = np.asarray(received_mens_email)
    visited = np.asarray(visited)
    fit_learner = LEARNER_FITTERS[learner_name]
    # Stratify on arm x outcome so every fold has visits in both arms.
    strata = 2 * received_mens_email + visited
    folds = StratifiedKFold(number_of_folds, shuffle=True, random_state=random_seed)
    out_of_fold_cate = np.empty(len(visited))
    for fold_index, (training_rows, held_out_rows) in enumerate(folds.split(covariates, strata)):
        learner = fit_learner(
            covariates.iloc[training_rows], received_mens_email[training_rows],
            visited[training_rows], min_samples_leaf, random_seed + 10 * fold_index,
        )
        out_of_fold_cate[held_out_rows] = learner.predict_cate(covariates.iloc[held_out_rows])
    return out_of_fold_cate


def cate_risk(received_mens_email, visited, estimated_cate):
    """Transformed-outcome loss mean((Z_i - tau_hat(X_i))^2).

    Since E[Z | X] = tau(X), its expectation equals E[(tau(X) - tau_hat(X))^2] plus a
    constant that does not depend on tau_hat, so smaller is better for ranking models.
    """
    z = transformed_outcome(received_mens_email, visited)
    return float(np.mean((z - estimated_cate) ** 2))


def aipw_scores(mu_mens, mu_womens, received_mens_email, visited, propensity=PROPENSITY_MENS_EMAIL):
    """Doubly robust (AIPW) scores Gamma_i(t) for each arm:

        Gamma_i(1) = mu_hat_1(X_i) + T_i       / e       * (Y_i - mu_hat_1(X_i))
        Gamma_i(0) = mu_hat_0(X_i) + (1 - T_i) / (1 - e) * (Y_i - mu_hat_0(X_i))

    Because e is known, E[Gamma_i(t)] = E[Y_i(t)] for ANY outcome model mu_hat fit on
    other data; a good mu_hat only shrinks the variance.
    """
    received_mens_email = np.asarray(received_mens_email, dtype=float)
    visited = np.asarray(visited, dtype=float)
    gamma_mens = mu_mens + received_mens_email / propensity * (visited - mu_mens)
    gamma_womens = mu_womens + (1 - received_mens_email) / (1 - propensity) * (visited - mu_womens)
    return gamma_mens, gamma_womens


def policy_scores(assign_mens_email, gamma_mens, gamma_womens):
    """Gamma_i(pi(X_i)): per-customer score of the email the rule would send."""
    assign_mens_email = np.asarray(assign_mens_email, dtype=bool)
    return np.where(assign_mens_email, gamma_mens, gamma_womens)


def mean_with_standard_error(per_customer_values):
    values = np.asarray(per_customer_values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def estimate_policy_value(assign_mens_email, gamma_mens, gamma_womens):
    """V_hat(pi) = (1/n) sum_i Gamma_i(pi(X_i)), with its CLT standard error."""
    return mean_with_standard_error(policy_scores(assign_mens_email, gamma_mens, gamma_womens))


def estimate_policy_value_difference(assign_mens_a, assign_mens_b, gamma_mens, gamma_womens):
    """V_hat(pi_a) - V_hat(pi_b) as a paired mean over the same customers.

    Pairing matters: only customers on whom the two rules disagree contribute, so the
    standard error is far smaller than if the two values were treated as independent.
    """
    paired_differences = (
        policy_scores(assign_mens_a, gamma_mens, gamma_womens)
        - policy_scores(assign_mens_b, gamma_mens, gamma_womens)
    )
    return mean_with_standard_error(paired_differences)


def group_average_treatment_effects(estimated_cate, gamma_mens, gamma_womens, number_of_groups=5):
    """Sort customers by tau_hat into quantile groups and estimate the true average
    Men's-minus-Women's effect within each group from the AIPW scores.

    If the forest found real heterogeneity, estimated effects should rise across groups.
    """
    group_index = pd.qcut(
        pd.Series(estimated_cate).rank(method="first"), number_of_groups, labels=False
    ).to_numpy()
    rows = []
    for group in range(number_of_groups):
        in_group = group_index == group
        effect, standard_error = mean_with_standard_error(gamma_mens[in_group] - gamma_womens[in_group])
        rows.append({
            "group": group + 1,
            "customers": int(in_group.sum()),
            "mean_predicted_effect": float(estimated_cate[in_group].mean()),
            "estimated_effect": effect,
            "standard_error": standard_error,
        })
    return pd.DataFrame(rows)
