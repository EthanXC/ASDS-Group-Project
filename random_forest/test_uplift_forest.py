"""Sanity checks for the data loader and the uplift / policy-evaluation functions.

Run from the repository root:  python -m unittest discover -s random_forest -v
"""

import unittest

import numpy as np
import pandas as pd

from hillstrom_data import (
    build_covariate_matrix, email_recipients_only, load_hillstrom,
)
from uplift_forest import (
    aipw_scores, estimate_policy_value, estimate_policy_value_difference,
    fit_two_forest_learner, fit_transformed_outcome_learner,
    group_average_treatment_effects, transformed_outcome,
)


def simulate_randomized_email_experiment(number_of_customers=40_000, random_seed=0):
    """Simulated experiment with a known CATE: the Men's email adds 10 pp for x1 = 1
    customers and subtracts 5 pp for x1 = 0 customers. Assignment is a fair coin."""
    generator = np.random.default_rng(random_seed)
    x1 = generator.integers(0, 2, number_of_customers)
    x2 = generator.normal(size=number_of_customers)  # pure noise covariate
    true_cate = np.where(x1 == 1, 0.10, -0.05)
    visit_probability_womens = 0.20 + 0.02 * x2.clip(-2, 2)
    visit_probability_mens = visit_probability_womens + true_cate
    potential_visit_womens = generator.random(number_of_customers) < visit_probability_womens
    potential_visit_mens = generator.random(number_of_customers) < visit_probability_mens
    received_mens_email = generator.integers(0, 2, number_of_customers)
    visited = np.where(received_mens_email == 1, potential_visit_mens, potential_visit_womens).astype(int)
    return {
        "covariates": pd.DataFrame({"x1": x1, "x2": x2}),
        "received_mens_email": received_mens_email,
        "visited": visited,
        "true_cate": true_cate,
        "visit_probability_mens": visit_probability_mens,
        "visit_probability_womens": visit_probability_womens,
    }


class TestTransformedOutcome(unittest.TestCase):
    def test_values_are_plus_or_minus_two_times_visit_when_propensity_is_half(self):
        z = transformed_outcome(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        np.testing.assert_allclose(z, [2.0, 0.0, -2.0, 0.0])

    def test_group_means_recover_the_true_effect(self):
        simulation = simulate_randomized_email_experiment()
        z = transformed_outcome(simulation["received_mens_email"], simulation["visited"])
        for x1_value, true_effect in [(1, 0.10), (0, -0.05)]:
            in_group = simulation["covariates"]["x1"].to_numpy() == x1_value
            self.assertAlmostEqual(z[in_group].mean(), true_effect, delta=0.025)


class TestAipwPolicyEvaluation(unittest.TestCase):
    def setUp(self):
        self.simulation = simulate_randomized_email_experiment()

    def test_aipw_is_unbiased_even_with_a_deliberately_wrong_outcome_model(self):
        n = len(self.simulation["visited"])
        wrong_constant_model = np.full(n, 0.5)
        gamma_mens, gamma_womens = aipw_scores(
            wrong_constant_model, wrong_constant_model,
            self.simulation["received_mens_email"], self.simulation["visited"],
        )
        self.assertAlmostEqual(gamma_mens.mean(), self.simulation["visit_probability_mens"].mean(), delta=0.01)
        self.assertAlmostEqual(gamma_womens.mean(), self.simulation["visit_probability_womens"].mean(), delta=0.01)

    def test_oracle_rule_value_matches_truth_and_beats_everyone_mens(self):
        gamma_mens, gamma_womens = aipw_scores(
            self.simulation["visit_probability_mens"], self.simulation["visit_probability_womens"],
            self.simulation["received_mens_email"], self.simulation["visited"],
        )
        oracle_rule = self.simulation["true_cate"] > 0
        true_oracle_value = np.where(
            oracle_rule, self.simulation["visit_probability_mens"], self.simulation["visit_probability_womens"]
        ).mean()
        estimated_value, standard_error = estimate_policy_value(oracle_rule, gamma_mens, gamma_womens)
        self.assertLess(abs(estimated_value - true_oracle_value), 3 * standard_error)

        everyone_mens = np.ones_like(oracle_rule)
        gain, gain_se = estimate_policy_value_difference(oracle_rule, everyone_mens, gamma_mens, gamma_womens)
        true_gain = 0.5 * 0.05  # half the customers gain 5 pp from switching to Women's
        self.assertLess(abs(gain - true_gain), 3 * gain_se)
        self.assertGreater(gain / gain_se, 3)

    def test_identical_rules_have_zero_difference_and_zero_standard_error(self):
        gamma_mens, gamma_womens = aipw_scores(
            np.full(4, 0.2), np.full(4, 0.2), np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])
        )
        rule = np.array([True, False, True, True])
        self.assertEqual(estimate_policy_value_difference(rule, rule, gamma_mens, gamma_womens), (0.0, 0.0))


class TestForestLearners(unittest.TestCase):
    def test_both_learners_recover_the_sign_of_a_strong_effect(self):
        simulation = simulate_randomized_email_experiment()
        covariates = simulation["covariates"]
        for fit_learner in [fit_two_forest_learner, fit_transformed_outcome_learner]:
            learner = fit_learner(covariates, simulation["received_mens_email"], simulation["visited"],
                                  min_samples_leaf=800, random_seed=0)
            predicted_sign_correct = (learner.predict_cate(covariates) > 0) == (simulation["true_cate"] > 0)
            self.assertGreater(predicted_sign_correct.mean(), 0.9, fit_learner.__name__)


class TestGroupAverageEffects(unittest.TestCase):
    def test_groups_partition_customers_and_follow_predicted_order(self):
        generator = np.random.default_rng(1)
        estimated_cate = generator.normal(size=1000)
        gamma_mens = estimated_cate + generator.normal(scale=0.1, size=1000)
        gamma_womens = np.zeros(1000)
        groups = group_average_treatment_effects(estimated_cate, gamma_mens, gamma_womens)
        self.assertEqual(groups["customers"].sum(), 1000)
        self.assertTrue(groups["mean_predicted_effect"].is_monotonic_increasing)
        self.assertTrue(groups["estimated_effect"].is_monotonic_increasing)


class TestHillstromData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers = load_hillstrom()

    def test_raw_data_shape_and_spelling_fix(self):
        self.assertEqual(self.customers.shape, (64_000, 12))
        self.assertNotIn("Surburban", set(self.customers["zip_code"]))
        self.assertEqual(set(self.customers["zip_code"]), {"Urban", "Suburban", "Rural"})

    def test_email_subset_drops_control_and_codes_treatment(self):
        email_recipients = email_recipients_only(self.customers)
        self.assertNotIn("No E-Mail", set(email_recipients["segment"]))
        self.assertEqual(len(email_recipients), (self.customers["segment"] != "No E-Mail").sum())
        mens_rows = email_recipients["segment"] == "Mens E-Mail"
        self.assertTrue((email_recipients.loc[mens_rows, "received_mens_email"] == 1).all())
        self.assertTrue((email_recipients.loc[~mens_rows, "received_mens_email"] == 0).all())

    def test_covariates_contain_no_outcomes_or_treatment(self):
        covariate_names = set(build_covariate_matrix(self.customers).columns)
        leaked = covariate_names & {"visit", "conversion", "spend", "segment", "received_mens_email"}
        self.assertEqual(leaked, set())
        self.assertNotIn("history_segment", covariate_names)


if __name__ == "__main__":
    unittest.main()
