#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Results Organizer
분석 결과를 구조화된 JSON 형태로 조직화하는 모듈
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class AnalysisResultsOrganizer:
    """Analysis results organizer for structured JSON output"""

    def __init__(self):
        self.results = {
            "metadata": {},
            "data_profile": {},
            "statistical_analysis": {},
            "data_quality": {},
            "insights": {},
            "recommendations": []
        }

    def set_metadata(self,
                     dataset_name: str,
                     analysis_timestamp: str = None,
                     analyst: str = "ML-MCP System",
                     version: str = "1.0") -> None:
        """Set analysis metadata"""

        self.results["metadata"] = {
            "dataset_name": dataset_name,
            "analysis_timestamp": analysis_timestamp or datetime.now().isoformat(),
            "analyst": analyst,
            "version": version,
            "analysis_duration": None
        }

    def set_data_profile(self, df: pd.DataFrame) -> None:
        """Set comprehensive data profile"""

        # Basic info
        basic_info = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "size_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "columns": list(df.columns)
        }

        # Column types analysis
        column_analysis = {}
        for col in df.columns:
            dtype = str(df[col].dtype)

            column_info = {
                "dtype": dtype,
                "category": self._categorize_column_type(df[col]),
                "unique_values": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                "memory_usage_bytes": int(df[col].memory_usage(deep=True))
            }

            # Add specific statistics based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                column_info.update(self._get_numeric_column_profile(df[col]))
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                column_info.update(self._get_categorical_column_profile(df[col]))
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_info.update(self._get_datetime_column_profile(df[col]))

            column_analysis[col] = column_info

        self.results["data_profile"] = {
            "basic_info": basic_info,
            "column_analysis": column_analysis,
            "data_types_summary": self._get_dtype_summary(df)
        }

    def add_statistical_analysis(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Add statistical analysis results"""

        if "statistical_analysis" not in self.results:
            self.results["statistical_analysis"] = {}

        # Organize results by analysis type
        organized_results = self._organize_analysis_results(analysis_type, results)
        self.results["statistical_analysis"][analysis_type] = organized_results

    def set_data_quality_assessment(self, df: pd.DataFrame) -> None:
        """Comprehensive data quality assessment"""

        quality_metrics = {
            "completeness": {
                "overall_completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                "complete_rows": int((~df.isnull().any(axis=1)).sum()),
                "complete_rows_percentage": round((~df.isnull().any(axis=1)).sum() / len(df) * 100, 2),
                "columns_with_missing": int((df.isnull().sum() > 0).sum())
            },
            "consistency": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percentage": round(df.duplicated().sum() / len(df) * 100, 2)
            },
            "validity": self._assess_data_validity(df),
            "uniqueness": self._assess_data_uniqueness(df)
        }

        # Quality score calculation
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_metrics["overall_quality_score"] = quality_score

        self.results["data_quality"] = quality_metrics

    def add_insights(self, insights: List[Dict[str, Any]]) -> None:
        """Add analytical insights"""

        categorized_insights = {
            "statistical_insights": [],
            "data_quality_insights": [],
            "pattern_insights": [],
            "anomaly_insights": []
        }

        for insight in insights:
            insight_type = insight.get("type", "general")
            category = self._categorize_insight(insight_type)
            categorized_insights[category].append(insight)

        self.results["insights"] = categorized_insights

    def add_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Add actionable recommendations"""

        prioritized_recommendations = sorted(
            recommendations,
            key=lambda x: x.get("priority", "medium") == "high"
        )

        self.results["recommendations"] = prioritized_recommendations

    def get_organized_results(self) -> Dict[str, Any]:
        """Get the final organized results"""
        return self.results

    def save_to_file(self, file_path: str) -> None:
        """Save organized results to JSON file"""

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

    def _categorize_column_type(self, series: pd.Series) -> str:
        """Categorize column type for better organization"""

        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                return "integer"
            else:
                return "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif series.dtype == 'category':
            return "categorical"
        else:
            # Check if it's likely categorical based on unique values
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 and series.nunique() < 50:
                return "categorical"
            else:
                return "text"

    def _get_numeric_column_profile(self, series: pd.Series) -> Dict[str, Any]:
        """Get detailed profile for numeric columns"""

        describe_stats = series.describe()

        return {
            "statistics": {
                "count": int(describe_stats['count']),
                "mean": float(describe_stats['mean']) if not pd.isna(describe_stats['mean']) else None,
                "std": float(describe_stats['std']) if not pd.isna(describe_stats['std']) else None,
                "min": float(describe_stats['min']) if not pd.isna(describe_stats['min']) else None,
                "q25": float(describe_stats['25%']) if not pd.isna(describe_stats['25%']) else None,
                "median": float(describe_stats['50%']) if not pd.isna(describe_stats['50%']) else None,
                "q75": float(describe_stats['75%']) if not pd.isna(describe_stats['75%']) else None,
                "max": float(describe_stats['max']) if not pd.isna(describe_stats['max']) else None
            },
            "distribution_info": {
                "skewness": float(series.skew()) if not pd.isna(series.skew()) else None,
                "kurtosis": float(series.kurtosis()) if not pd.isna(series.kurtosis()) else None,
                "zeros_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum()),
                "positive_count": int((series > 0).sum())
            }
        }

    def _get_categorical_column_profile(self, series: pd.Series) -> Dict[str, Any]:
        """Get detailed profile for categorical columns"""

        value_counts = series.value_counts()

        return {
            "categories": {
                "unique_categories": int(series.nunique()),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            },
            "frequency_distribution": value_counts.head(10).to_dict(),
            "text_properties": {
                "avg_length": round(series.astype(str).str.len().mean(), 2) if not series.empty else 0,
                "max_length": int(series.astype(str).str.len().max()) if not series.empty else 0,
                "min_length": int(series.astype(str).str.len().min()) if not series.empty else 0
            }
        }

    def _get_datetime_column_profile(self, series: pd.Series) -> Dict[str, Any]:
        """Get detailed profile for datetime columns"""

        return {
            "time_range": {
                "earliest": str(series.min()) if not pd.isna(series.min()) else None,
                "latest": str(series.max()) if not pd.isna(series.max()) else None,
                "span_days": int((series.max() - series.min()).days) if not pd.isna(series.min()) and not pd.isna(series.max()) else None
            },
            "time_patterns": {
                "year_range": [int(series.dt.year.min()), int(series.dt.year.max())] if not series.empty else None,
                "month_distribution": series.dt.month.value_counts().to_dict() if not series.empty else {},
                "day_of_week_distribution": series.dt.day_name().value_counts().to_dict() if not series.empty else {}
            }
        }

    def _get_dtype_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get summary of data types"""

        dtype_counts = {}
        for col in df.columns:
            category = self._categorize_column_type(df[col])
            dtype_counts[category] = dtype_counts.get(category, 0) + 1

        return dtype_counts

    def _organize_analysis_results(self, analysis_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize analysis results based on type"""

        if analysis_type == "descriptive_statistics":
            return self._organize_descriptive_stats(results)
        elif analysis_type == "correlation_analysis":
            return self._organize_correlation_analysis(results)
        elif analysis_type == "missing_data_analysis":
            return self._organize_missing_data_analysis(results)
        elif analysis_type == "data_types_analysis":
            return self._organize_data_types_analysis(results)
        else:
            return results

    def _organize_descriptive_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize descriptive statistics results"""

        organized = {
            "summary": results.get("data_summary", {}),
            "numeric_analysis": {
                "columns": results.get("numeric_columns", []),
                "statistics": results.get("numeric_statistics", {})
            },
            "categorical_analysis": {
                "columns": results.get("categorical_columns", []),
                "statistics": results.get("categorical_statistics", {})
            }
        }

        return organized

    def _organize_correlation_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize correlation analysis results"""

        return {
            "data_info": results.get("data_info", {}),
            "correlation_matrix": results.get("correlation_matrix", {}),
            "significant_correlations": {
                "strong_correlations": results.get("strong_correlations", []),
                "correlation_insights": results.get("insights", [])
            }
        }

    def _organize_missing_data_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize missing data analysis results"""

        return {
            "overview": results.get("missing_data_summary", {}),
            "column_breakdown": results.get("missing_by_column", {}),
            "recommendations": results.get("recommendations", [])
        }

    def _organize_data_types_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize data types analysis results"""

        return {
            "column_analysis": results.get("column_analysis", {}),
            "optimization_potential": results.get("memory_optimization_potential", {}),
            "type_recommendations": self._extract_type_recommendations(results)
        }

    def _extract_type_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract type optimization recommendations"""

        recommendations = []
        column_analysis = results.get("column_analysis", {})

        for col, analysis in column_analysis.items():
            if "optimization" in analysis:
                recommendations.append({
                    "column": col,
                    "current_type": analysis.get("current_type"),
                    "recommendation": analysis.get("optimization"),
                    "potential_memory_saving": analysis.get("memory_usage", 0)
                })

        return recommendations

    def _assess_data_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity"""

        validity_issues = []

        # Check for common validity issues
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validity_issues.append({
                        "column": col,
                        "issue": "infinite_values",
                        "count": int(inf_count)
                    })

                # Check for extreme outliers (beyond 3 standard deviations)
                if df[col].std() > 0:
                    outliers = ((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum()
                    if outliers > 0:
                        validity_issues.append({
                            "column": col,
                            "issue": "extreme_outliers",
                            "count": int(outliers)
                        })

        return {
            "validity_issues": validity_issues,
            "validity_score": max(0, 100 - len(validity_issues) * 10)  # Simple scoring
        }

    def _assess_data_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""

        uniqueness_analysis = {}

        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            uniqueness_analysis[col] = {
                "unique_count": int(df[col].nunique()),
                "unique_ratio": round(unique_ratio, 4),
                "is_unique_identifier": unique_ratio == 1.0,
                "is_highly_unique": unique_ratio > 0.95,
                "is_low_cardinality": unique_ratio < 0.1
            }

        return uniqueness_analysis

    def _calculate_quality_score(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall data quality score"""

        # Weights for different quality aspects
        weights = {
            "completeness": 0.3,
            "consistency": 0.2,
            "validity": 0.3,
            "uniqueness": 0.2
        }

        # Calculate component scores
        completeness_score = quality_metrics["completeness"]["overall_completeness"]
        consistency_score = max(0, 100 - quality_metrics["consistency"]["duplicate_percentage"])
        validity_score = quality_metrics["validity"]["validity_score"]

        # Simplified uniqueness score (avoiding too many unique or too few unique columns)
        uniqueness_scores = []
        for col_analysis in quality_metrics["uniqueness"].values():
            if col_analysis["is_unique_identifier"]:
                uniqueness_scores.append(100)
            elif col_analysis["is_highly_unique"] or col_analysis["is_low_cardinality"]:
                uniqueness_scores.append(80)
            else:
                uniqueness_scores.append(90)

        uniqueness_score = np.mean(uniqueness_scores) if uniqueness_scores else 50

        # Weighted overall score
        overall_score = (
            completeness_score * weights["completeness"] +
            consistency_score * weights["consistency"] +
            validity_score * weights["validity"] +
            uniqueness_score * weights["uniqueness"]
        )

        return {
            "overall_score": round(overall_score, 1),
            "component_scores": {
                "completeness": round(completeness_score, 1),
                "consistency": round(consistency_score, 1),
                "validity": round(validity_score, 1),
                "uniqueness": round(uniqueness_score, 1)
            },
            "grade": self._score_to_grade(overall_score)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""

        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _categorize_insight(self, insight_type: str) -> str:
        """Categorize insights for better organization"""

        insight_categories = {
            "statistical_insights": ["correlation", "distribution", "central_tendency"],
            "data_quality_insights": ["missing_data", "duplicates", "outliers"],
            "pattern_insights": ["trend", "seasonality", "pattern"],
            "anomaly_insights": ["anomaly", "outlier", "unusual"]
        }

        for category, keywords in insight_categories.items():
            if any(keyword in insight_type.lower() for keyword in keywords):
                return category

        return "statistical_insights"  # Default category