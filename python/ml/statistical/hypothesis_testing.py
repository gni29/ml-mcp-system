#!/usr/bin/env python3
"""
Hypothesis Testing Module for ML MCP System
Statistical hypothesis tests for data analysis
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp,
    mannwhitneyu, wilcoxon,
    chi2_contingency, fisher_exact,
    f_oneway, kruskal,
    pearsonr, spearmanr, kendalltau,
    shapiro, normaltest, kstest,
    levene, bartlett
)


class HypothesisTest:
    """Perform statistical hypothesis tests"""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize hypothesis tester

        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha

    def t_test_independent(self, group1: np.ndarray, group2: np.ndarray,
                          equal_var: bool = True) -> Dict[str, Any]:
        """
        Independent samples t-test

        Args:
            group1: First group data
            group2: Second group data
            equal_var: Assume equal variance

        Returns:
            Test results
        """
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var)

        return {
            'test': 'independent_t_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'equal_var': equal_var,
            'interpretation': self._interpret_result(p_value, 'difference between groups')
        }

    def t_test_paired(self, before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        """
        Paired samples t-test

        Args:
            before: Measurements before treatment
            after: Measurements after treatment

        Returns:
            Test results
        """
        statistic, p_value = ttest_rel(before, after)

        return {
            'test': 'paired_t_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'mean_difference': float(np.mean(after - before)),
            'interpretation': self._interpret_result(p_value, 'change before/after')
        }

    def t_test_one_sample(self, data: np.ndarray, population_mean: float) -> Dict[str, Any]:
        """
        One sample t-test

        Args:
            data: Sample data
            population_mean: Expected population mean

        Returns:
            Test results
        """
        statistic, p_value = ttest_1samp(data, population_mean)

        return {
            'test': 'one_sample_t_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'sample_mean': float(np.mean(data)),
            'population_mean': float(population_mean),
            'interpretation': self._interpret_result(p_value, f'difference from {population_mean}')
        }

    def mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Mann-Whitney U test (non-parametric alternative to t-test)

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            Test results
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        return {
            'test': 'mann_whitney_u',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'interpretation': self._interpret_result(p_value, 'difference in distributions')
        }

    def wilcoxon_test(self, before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test (non-parametric paired test)

        Args:
            before: Measurements before
            after: Measurements after

        Returns:
            Test results
        """
        statistic, p_value = wilcoxon(before, after)

        return {
            'test': 'wilcoxon_signed_rank',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'interpretation': self._interpret_result(p_value, 'median difference')
        }

    def chi_square_test(self, contingency_table: np.ndarray) -> Dict[str, Any]:
        """
        Chi-square test of independence

        Args:
            contingency_table: 2D contingency table

        Returns:
            Test results
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        return {
            'test': 'chi_square_independence',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'interpretation': self._interpret_result(p_value, 'association between variables')
        }

    def fisher_exact_test(self, contingency_table: np.ndarray) -> Dict[str, Any]:
        """
        Fisher's exact test (for 2x2 tables)

        Args:
            contingency_table: 2x2 contingency table

        Returns:
            Test results
        """
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires 2x2 table")

        odds_ratio, p_value = fisher_exact(contingency_table)

        return {
            'test': 'fisher_exact',
            'odds_ratio': float(odds_ratio),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'interpretation': self._interpret_result(p_value, 'association')
        }

    def anova_one_way(self, *groups) -> Dict[str, Any]:
        """
        One-way ANOVA test

        Args:
            *groups: Multiple group arrays

        Returns:
            Test results
        """
        f_statistic, p_value = f_oneway(*groups)

        return {
            'test': 'one_way_anova',
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'num_groups': len(groups),
            'interpretation': self._interpret_result(p_value, 'difference among groups')
        }

    def kruskal_wallis(self, *groups) -> Dict[str, Any]:
        """
        Kruskal-Wallis H test (non-parametric ANOVA)

        Args:
            *groups: Multiple group arrays

        Returns:
            Test results
        """
        h_statistic, p_value = kruskal(*groups)

        return {
            'test': 'kruskal_wallis',
            'h_statistic': float(h_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'num_groups': len(groups),
            'interpretation': self._interpret_result(p_value, 'difference in distributions')
        }

    def correlation_test(self, x: np.ndarray, y: np.ndarray,
                        method: str = 'pearson') -> Dict[str, Any]:
        """
        Test correlation significance

        Args:
            x: First variable
            y: Second variable
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Test results
        """
        if method == 'pearson':
            corr, p_value = pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = kendalltau(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'test': f'{method}_correlation',
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'interpretation': self._interpret_result(p_value, 'correlation')
        }

    def normality_test(self, data: np.ndarray, method: str = 'shapiro') -> Dict[str, Any]:
        """
        Test for normality

        Args:
            data: Data to test
            method: 'shapiro', 'normaltest', or 'ks'

        Returns:
            Test results
        """
        if method == 'shapiro':
            statistic, p_value = shapiro(data)
            test_name = 'shapiro_wilk'
        elif method == 'normaltest':
            statistic, p_value = normaltest(data)
            test_name = 'dagostino_pearson'
        elif method == 'ks':
            # Kolmogorov-Smirnov test against normal distribution
            statistic, p_value = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            test_name = 'kolmogorov_smirnov'
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'test': f'{test_name}_normality',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'normal': p_value >= self.alpha,  # Fail to reject null = normal
            'alpha': self.alpha,
            'interpretation': 'Data appears normally distributed' if p_value >= self.alpha
                            else 'Data significantly deviates from normality'
        }

    def variance_equality_test(self, *groups, method: str = 'levene') -> Dict[str, Any]:
        """
        Test equality of variances

        Args:
            *groups: Multiple group arrays
            method: 'levene' or 'bartlett'

        Returns:
            Test results
        """
        if method == 'levene':
            statistic, p_value = levene(*groups)
            test_name = 'levene'
        elif method == 'bartlett':
            statistic, p_value = bartlett(*groups)
            test_name = 'bartlett'
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'test': f'{test_name}_variance_equality',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'equal_variance': p_value >= self.alpha,
            'alpha': self.alpha,
            'num_groups': len(groups),
            'interpretation': 'Variances are equal' if p_value >= self.alpha
                            else 'Variances are significantly different'
        }

    def _interpret_result(self, p_value: float, effect: str) -> str:
        """Generate interpretation text"""
        if p_value < self.alpha:
            return f"Significant {effect} detected (p={p_value:.4f} < {self.alpha})"
        else:
            return f"No significant {effect} (p={p_value:.4f} >= {self.alpha})"


class MultipleComparison:
    """Handle multiple comparison corrections"""

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Bonferroni correction for multiple comparisons

        Args:
            p_values: List of p-values
            alpha: Original significance level

        Returns:
            Corrected results
        """
        n = len(p_values)
        adjusted_alpha = alpha / n
        significant = [p < adjusted_alpha for p in p_values]

        return {
            'method': 'bonferroni',
            'original_alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'num_tests': n,
            'p_values': p_values,
            'significant': significant,
            'num_significant': sum(significant)
        }

    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        False Discovery Rate (Benjamini-Hochberg) correction

        Args:
            p_values: List of p-values
            alpha: Original significance level

        Returns:
            Corrected results
        """
        from statsmodels.stats.multitest import multipletests

        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

        return {
            'method': 'fdr_benjamini_hochberg',
            'original_alpha': alpha,
            'num_tests': len(p_values),
            'p_values': p_values,
            'p_corrected': p_corrected.tolist(),
            'significant': reject.tolist(),
            'num_significant': sum(reject)
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python hypothesis_testing.py <test_type> <args...>")
        print("Test types: t_test, mann_whitney, chi_square, anova, correlation, normality")
        sys.exit(1)

    test_type = sys.argv[1]

    try:
        tester = HypothesisTest(alpha=0.05)

        if test_type == 'demo':
            # Demo with synthetic data
            np.random.seed(42)
            group1 = np.random.normal(100, 15, 50)
            group2 = np.random.normal(105, 15, 50)

            results = {
                't_test': tester.t_test_independent(group1, group2),
                'mann_whitney': tester.mann_whitney_u(group1, group2),
                'normality_group1': tester.normality_test(group1),
                'variance_test': tester.variance_equality_test(group1, group2)
            }

            # Convert numpy bools to Python bools
            def convert_bools(obj):
                if isinstance(obj, dict):
                    return {k: convert_bools(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bools(item) for item in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return obj

            results = convert_bools(results)

        else:
            results = {'error': 'Please use demo mode or provide data file'}

        print(json.dumps(results, ensure_ascii=False, indent=2))

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