#!/usr/bin/env python3
"""
Bayesian Analysis Module for ML MCP System
Bayesian statistical inference and modeling
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class BayesianInference:
    """Bayesian statistical inference"""

    def __init__(self):
        """Initialize Bayesian inference"""
        pass

    def bayes_theorem(self, prior: float, likelihood: float, evidence: float) -> float:
        """
        Apply Bayes' theorem

        P(A|B) = P(B|A) * P(A) / P(B)

        Args:
            prior: P(A) - Prior probability
            likelihood: P(B|A) - Likelihood
            evidence: P(B) - Evidence

        Returns:
            Posterior probability P(A|B)
        """
        posterior = (likelihood * prior) / evidence
        return posterior

    def beta_binomial_update(self, alpha_prior: float, beta_prior: float,
                             successes: int, trials: int) -> Dict[str, Any]:
        """
        Beta-Binomial conjugate update

        Args:
            alpha_prior: Prior alpha parameter
            beta_prior: Prior beta parameter
            successes: Number of successes observed
            trials: Number of trials

        Returns:
            Posterior distribution parameters
        """
        failures = trials - successes

        # Posterior parameters
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + failures

        # Prior and posterior means
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        post_mean = alpha_post / (alpha_post + beta_post)

        # Credible interval (95%)
        lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        upper = stats.beta.ppf(0.975, alpha_post, beta_post)

        return {
            'prior': {
                'alpha': alpha_prior,
                'beta': beta_prior,
                'mean': prior_mean
            },
            'data': {
                'successes': successes,
                'failures': failures,
                'trials': trials
            },
            'posterior': {
                'alpha': alpha_post,
                'beta': beta_post,
                'mean': post_mean,
                'credible_interval_95': [lower, upper]
            }
        }

    def normal_normal_update(self, prior_mean: float, prior_std: float,
                            data_mean: float, data_std: float,
                            n_observations: int) -> Dict[str, Any]:
        """
        Normal-Normal conjugate update

        Args:
            prior_mean: Prior mean
            prior_std: Prior standard deviation
            data_mean: Sample mean
            data_std: Sample standard deviation
            n_observations: Number of observations

        Returns:
            Posterior distribution parameters
        """
        # Prior precision
        prior_precision = 1 / (prior_std ** 2)

        # Data precision
        data_precision = n_observations / (data_std ** 2)

        # Posterior precision
        post_precision = prior_precision + data_precision

        # Posterior mean
        post_mean = (prior_precision * prior_mean + data_precision * data_mean) / post_precision

        # Posterior std
        post_std = np.sqrt(1 / post_precision)

        # Credible interval (95%)
        lower = stats.norm.ppf(0.025, post_mean, post_std)
        upper = stats.norm.ppf(0.975, post_mean, post_std)

        return {
            'prior': {
                'mean': prior_mean,
                'std': prior_std
            },
            'data': {
                'mean': data_mean,
                'std': data_std,
                'n': n_observations
            },
            'posterior': {
                'mean': post_mean,
                'std': post_std,
                'credible_interval_95': [lower, upper]
            }
        }

    def ab_test_bayesian(self, conversions_a: int, trials_a: int,
                        conversions_b: int, trials_b: int,
                        prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Dict[str, Any]:
        """
        Bayesian A/B test

        Args:
            conversions_a: Conversions in variant A
            trials_a: Trials in variant A
            conversions_b: Conversions in variant B
            trials_b: Trials in variant B
            prior_alpha: Prior alpha
            prior_beta: Prior beta

        Returns:
            A/B test results
        """
        # Posterior for A
        alpha_a = prior_alpha + conversions_a
        beta_a = prior_beta + (trials_a - conversions_a)

        # Posterior for B
        alpha_b = prior_alpha + conversions_b
        beta_b = prior_beta + (trials_b - conversions_b)

        # Sample from posteriors
        n_samples = 100000
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)

        # Probability B > A
        prob_b_better = np.mean(samples_b > samples_a)

        # Expected loss if choosing wrong variant
        loss_choosing_a = np.mean(np.maximum(samples_b - samples_a, 0))
        loss_choosing_b = np.mean(np.maximum(samples_a - samples_b, 0))

        return {
            'variant_a': {
                'conversions': conversions_a,
                'trials': trials_a,
                'conversion_rate': conversions_a / trials_a,
                'posterior': {
                    'alpha': alpha_a,
                    'beta': beta_a,
                    'mean': alpha_a / (alpha_a + beta_a)
                }
            },
            'variant_b': {
                'conversions': conversions_b,
                'trials': trials_b,
                'conversion_rate': conversions_b / trials_b,
                'posterior': {
                    'alpha': alpha_b,
                    'beta': beta_b,
                    'mean': alpha_b / (alpha_b + beta_b)
                }
            },
            'comparison': {
                'prob_b_better_than_a': prob_b_better,
                'prob_a_better_than_b': 1 - prob_b_better,
                'expected_loss_choosing_a': loss_choosing_a,
                'expected_loss_choosing_b': loss_choosing_b,
                'recommendation': 'B' if prob_b_better > 0.95 else ('A' if prob_b_better < 0.05 else 'Inconclusive')
            }
        }


class NaiveBayesClassifier:
    """Naive Bayes classifier wrapper"""

    def __init__(self, variant: str = 'gaussian'):
        """
        Initialize Naive Bayes classifier

        Args:
            variant: 'gaussian', 'multinomial', or 'bernoulli'
        """
        self.variant = variant

        if variant == 'gaussian':
            self.model = GaussianNB()
        elif variant == 'multinomial':
            self.model = MultinomialNB()
        elif variant == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def train(self, X: pd.DataFrame, y: pd.Series,
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train Naive Bayes classifier

        Args:
            X: Features
            y: Target
            test_size: Test set proportion

        Returns:
            Training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'model': f'naive_bayes_{self.variant}',
            'accuracy': float(accuracy),
            'classes': self.model.classes_.tolist(),
            'num_features': X.shape[1],
            'test_samples': len(y_test)
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


class BayesianEstimator:
    """Bayesian parameter estimation"""

    @staticmethod
    def estimate_mean(data: np.ndarray, prior_mean: float = 0,
                     prior_std: float = 1000) -> Dict[str, Any]:
        """
        Estimate mean with Bayesian inference

        Args:
            data: Observed data
            prior_mean: Prior mean
            prior_std: Prior standard deviation

        Returns:
            Posterior estimates
        """
        n = len(data)
        data_mean = np.mean(data)
        data_std = np.std(data, ddof=1)

        # Update
        prior_precision = 1 / (prior_std ** 2)
        data_precision = n / (data_std ** 2)
        post_precision = prior_precision + data_precision

        post_mean = (prior_precision * prior_mean + data_precision * data_mean) / post_precision
        post_std = np.sqrt(1 / post_precision)

        # Credible interval
        lower = stats.norm.ppf(0.025, post_mean, post_std)
        upper = stats.norm.ppf(0.975, post_mean, post_std)

        return {
            'posterior_mean': float(post_mean),
            'posterior_std': float(post_std),
            'credible_interval_95': [float(lower), float(upper)],
            'data_mean': float(data_mean),
            'n_observations': n
        }

    @staticmethod
    def estimate_proportion(successes: int, trials: int,
                           prior_alpha: float = 1, prior_beta: float = 1) -> Dict[str, Any]:
        """
        Estimate proportion with Bayesian inference

        Args:
            successes: Number of successes
            trials: Number of trials
            prior_alpha: Prior alpha
            prior_beta: Prior beta

        Returns:
            Posterior estimates
        """
        failures = trials - successes

        alpha_post = prior_alpha + successes
        beta_post = prior_beta + failures

        post_mean = alpha_post / (alpha_post + beta_post)

        # Credible interval
        lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        upper = stats.beta.ppf(0.975, alpha_post, beta_post)

        return {
            'posterior_mean': float(post_mean),
            'credible_interval_95': [float(lower), float(upper)],
            'observed_proportion': successes / trials,
            'successes': successes,
            'trials': trials
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python bayesian_analysis.py <analysis_type>")
        print("Types: ab_test, beta_update, normal_update, estimate_mean, estimate_proportion")
        sys.exit(1)

    analysis_type = sys.argv[1]

    try:
        if analysis_type == 'ab_test':
            # Demo A/B test
            bayesian = BayesianInference()
            results = bayesian.ab_test_bayesian(
                conversions_a=450, trials_a=5000,
                conversions_b=480, trials_b=5000
            )

        elif analysis_type == 'beta_update':
            bayesian = BayesianInference()
            results = bayesian.beta_binomial_update(
                alpha_prior=1, beta_prior=1,
                successes=45, trials=100
            )

        elif analysis_type == 'estimate_mean':
            np.random.seed(42)
            data = np.random.normal(100, 15, 50)
            estimator = BayesianEstimator()
            results = estimator.estimate_mean(data)

        else:
            results = {'error': f'Unknown analysis type: {analysis_type}'}

        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()