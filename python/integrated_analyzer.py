#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Integrated Data Analyzer
향상된 통합 데이터 분석기 - 상세한 JSON 결과와 고급 HTML 보고서를 동시에 생성
"""

import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
try:
    import sys
    from pathlib import Path

    # Add the current directory to Python path for imports
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from utils.json_organizer import AnalysisResultsOrganizer
    from utils.enhanced_html_generator import EnhancedHTMLReportGenerator
    HAS_ENHANCED_MODULES = True
    print("Enhanced modules loaded successfully")
except ImportError as e:
    HAS_ENHANCED_MODULES = False
    print(f"Enhanced modules not found: {e}, using basic functionality")

class IntegratedAnalyzer:
    """Enhanced integrated analyzer that produces detailed JSON results and advanced HTML reports"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_results = []

        # Initialize enhanced components if available
        if HAS_ENHANCED_MODULES:
            self.json_organizer = AnalysisResultsOrganizer()
            self.html_generator = EnhancedHTMLReportGenerator()
        else:
            self.json_organizer = None
            self.html_generator = None

    def analyze_data(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """Perform complete enhanced data analysis and generate outputs"""

        print("Starting enhanced integrated data analysis...")

        # Initialize enhanced JSON organizer if available
        if self.json_organizer:
            self.json_organizer.set_metadata(dataset_name)
            self.json_organizer.set_data_profile(df)
            self.json_organizer.set_data_quality_assessment(df)

        # 1. Basic Statistics Analysis
        print("Running descriptive statistics analysis...")
        stats_result = self._analyze_descriptive_statistics(df)
        self.analysis_results.append(stats_result)

        if self.json_organizer:
            self.json_organizer.add_statistical_analysis("descriptive_statistics", stats_result)

        # 2. Correlation Analysis (if applicable)
        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
            print("Running correlation analysis...")
            corr_result = self._analyze_correlations(df)
            self.analysis_results.append(corr_result)

            if self.json_organizer:
                self.json_organizer.add_statistical_analysis("correlation_analysis", corr_result)

        # 3. Missing Data Analysis
        print("Running missing data analysis...")
        missing_result = self._analyze_missing_data(df)
        self.analysis_results.append(missing_result)

        if self.json_organizer:
            self.json_organizer.add_statistical_analysis("missing_data_analysis", missing_result)

        # 4. Data Types Analysis
        print("Running data types analysis...")
        types_result = self._analyze_data_types(df)
        self.analysis_results.append(types_result)

        if self.json_organizer:
            self.json_organizer.add_statistical_analysis("data_types_analysis", types_result)

        # 5. Generate Enhanced HTML Report
        print("Generating enhanced HTML report...")
        html_report_path = self._generate_enhanced_html_report(df, dataset_name)

        # 6. Save Enhanced JSON Results
        json_report_path = self._save_json_results()

        return {
            "success": True,
            "analysis_completed": True,
            "total_analyses": len(self.analysis_results),
            "reports_generated": {
                "html_report": html_report_path,
                "json_report": json_report_path
            },
            "data_overview": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns)
            },
            "analysis_summary": self._create_analysis_summary()
        }

    def _analyze_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze descriptive statistics"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        result = {
            "analysis_type": "descriptive_statistics",
            "success": True,
            "data_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols)
            }
        }

        # Numeric statistics
        if numeric_cols:
            numeric_stats = {}
            for stat in ['count', 'mean', 'median', 'std', 'min', 'max']:
                numeric_stats[stat] = {}
                for col in numeric_cols:
                    if stat == 'count':
                        numeric_stats[stat][col] = df[col].count()
                    elif stat == 'mean':
                        numeric_stats[stat][col] = float(df[col].mean())
                    elif stat == 'median':
                        numeric_stats[stat][col] = float(df[col].median())
                    elif stat == 'std':
                        numeric_stats[stat][col] = float(df[col].std())
                    elif stat == 'min':
                        numeric_stats[stat][col] = float(df[col].min())
                    elif stat == 'max':
                        numeric_stats[stat][col] = float(df[col].max())

            result["numeric_columns"] = numeric_cols
            result["numeric_statistics"] = numeric_stats

            # Add quartiles
            q25_dict = {}
            q75_dict = {}
            for col in numeric_cols:
                q25_dict[col] = float(df[col].quantile(0.25))
                q75_dict[col] = float(df[col].quantile(0.75))

            result["numeric_statistics"]["q25"] = q25_dict
            result["numeric_statistics"]["q75"] = q75_dict

        # Categorical statistics
        if categorical_cols:
            categorical_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                categorical_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "null_count": df[col].isnull().sum(),
                    "non_null_count": df[col].count(),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "frequency_table": value_counts.head(10).to_dict()  # Top 10 only
                }

            result["categorical_columns"] = categorical_cols
            result["categorical_statistics"] = categorical_stats

        return result

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return {
                "analysis_type": "correlation_analysis",
                "success": False,
                "error": "Need at least 2 numeric columns for correlation analysis"
            }

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= 0.3:  # Threshold for "interesting" correlations
                    strength = self._classify_correlation_strength(abs(corr_value))
                    strong_correlations.append({
                        "variable1": col1,
                        "variable2": col2,
                        "correlation": round(float(corr_value), 4),
                        "strength": strength,
                        "direction": "positive" if corr_value > 0 else "negative"
                    })

        # Sort by correlation strength
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "analysis_type": "correlation_analysis",
            "success": True,
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": numeric_cols,
                "numeric_column_count": len(numeric_cols)
            },
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "insights": self._generate_correlation_insights(strong_correlations)
        }

    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength"""
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _generate_correlation_insights(self, correlations: List[Dict]) -> List[Dict]:
        """Generate insights from correlation analysis"""
        insights = []

        if not correlations:
            insights.append({
                "type": "no_correlations",
                "message": "No significant correlations found between variables",
                "severity": "info"
            })
            return insights

        # Very strong correlations
        very_strong = [c for c in correlations if c["strength"] == "very_strong"]
        if very_strong:
            insights.append({
                "type": "multicollinearity_warning",
                "message": f"{len(very_strong)}개의 매우 강한 상관관계가 발견되었습니다. 다중공선성을 주의하세요.",
                "severity": "warning"
            })

        # Highlight top correlations
        for corr in correlations[:3]:  # Top 3
            direction = "양의" if corr["direction"] == "positive" else "음의"
            insights.append({
                "type": "strong_relationship",
                "message": f"'{corr['variable1']}'와 '{corr['variable2']}' 간 {corr['strength']} {direction} 상관관계 (r={corr['correlation']})",
                "severity": "info"
            })

        return insights

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""

        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100

        columns_with_missing = missing_counts[missing_counts > 0].to_dict()

        return {
            "analysis_type": "missing_data_analysis",
            "success": True,
            "missing_data_summary": {
                "total_missing_values": int(total_missing),
                "missing_percentage": round(missing_percentage, 2),
                "columns_with_missing": len(columns_with_missing),
                "complete_rows": len(df.dropna())
            },
            "missing_by_column": {k: int(v) for k, v in columns_with_missing.items()},
            "recommendations": self._generate_missing_data_recommendations(df, columns_with_missing)
        }

    def _generate_missing_data_recommendations(self, df: pd.DataFrame, missing_cols: Dict) -> List[Dict]:
        """Generate recommendations for handling missing data"""
        recommendations = []

        if not missing_cols:
            recommendations.append({
                "type": "no_action_needed",
                "message": "축하합니다! 결측치가 없는 완전한 데이터셋입니다.",
                "action": "추가 조치 불필요"
            })
            return recommendations

        total_rows = len(df)

        for col, missing_count in missing_cols.items():
            missing_pct = (missing_count / total_rows) * 100

            if missing_pct > 50:
                recommendations.append({
                    "type": "high_missing",
                    "column": col,
                    "message": f"{col} 열은 {missing_pct:.1f}%의 높은 결측치를 가지고 있습니다.",
                    "action": "열 제거를 고려하거나 신중한 대체 방법 필요"
                })
            elif missing_pct > 20:
                recommendations.append({
                    "type": "moderate_missing",
                    "column": col,
                    "message": f"{col} 열은 {missing_pct:.1f}%의 중간 수준 결측치를 가지고 있습니다.",
                    "action": "적절한 대체 방법 적용 (평균, 중위수, 최빈값)"
                })
            else:
                recommendations.append({
                    "type": "low_missing",
                    "column": col,
                    "message": f"{col} 열은 {missing_pct:.1f}%의 낮은 결측치를 가지고 있습니다.",
                    "action": "단순 대체 또는 제거 가능"
                })

        return recommendations

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and suggest improvements"""

        column_analysis = {}

        for col in df.columns:
            col_data = df[col]
            analysis = {
                "current_type": str(col_data.dtype),
                "unique_count": col_data.nunique(),
                "null_count": col_data.isnull().sum(),
                "memory_usage": col_data.memory_usage(deep=True)
            }

            # Suggest optimizations
            if pd.api.types.is_numeric_dtype(col_data):
                analysis["category"] = "numeric"
                if col_data.dtype == 'int64' and col_data.min() >= 0 and col_data.max() < 255:
                    analysis["optimization"] = "Convert to uint8 to save memory"
                elif col_data.dtype == 'float64' and not col_data.isnull().any():
                    analysis["optimization"] = "Consider float32 if precision allows"
            elif pd.api.types.is_object_dtype(col_data):
                analysis["category"] = "text/categorical"
                if col_data.nunique() < len(col_data) * 0.5:
                    analysis["optimization"] = "Convert to category type to save memory"
                else:
                    analysis["optimization"] = "Consider string compression"

            column_analysis[col] = analysis

        return {
            "analysis_type": "data_types_analysis",
            "success": True,
            "column_analysis": column_analysis,
            "memory_optimization_potential": self._calculate_memory_savings(column_analysis)
        }

    def _calculate_memory_savings(self, column_analysis: Dict) -> Dict[str, Any]:
        """Calculate potential memory savings"""
        total_current = sum(col["memory_usage"] for col in column_analysis.values())

        return {
            "current_memory_bytes": total_current,
            "current_memory_mb": round(total_current / (1024**2), 2),
            "optimization_available": any("optimization" in col for col in column_analysis.values())
        }

    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML report"""

        from datetime import datetime

        # Create simplified HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; margin: -30px -30px 30px -30px; border-radius: 10px 10px 0 0; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ text-align: center; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .label {{ color: #666; margin-top: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .highlight {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by ML MCP System</p>
        </div>

        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{len(df):,}</div>
                    <div class="label">Total Rows</div>
                </div>
                <div class="stat-card">
                    <div class="number">{len(df.columns)}</div>
                    <div class="label">Total Columns</div>
                </div>
                <div class="stat-card">
                    <div class="number">{len(df.select_dtypes(include=[np.number]).columns)}</div>
                    <div class="label">Numeric Columns</div>
                </div>
                <div class="stat-card">
                    <div class="number">{df.isnull().sum().sum()}</div>
                    <div class="label">Missing Values</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Column Information</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Unique Values</th>
                        <th>Missing Count</th>
                        <th>Missing %</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add column information
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            html_content += f"""
                    <tr>
                        <td><strong>{col}</strong></td>
                        <td>{df[col].dtype}</td>
                        <td>{df[col].nunique():,}</td>
                        <td>{missing_count}</td>
                        <td>{missing_pct:.1f}%</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Analysis Summary</h2>
"""

        # Add analysis results summary
        for result in self.analysis_results:
            analysis_type = result.get("analysis_type", "Unknown").replace("_", " ").title()
            html_content += f"""
            <div class="highlight">
                <h3>{analysis_type}</h3>
                <p>Analysis completed successfully. Check the JSON report for detailed results.</p>
            </div>
"""

        html_content += """
        </div>

        <div class="section">
            <h2>Sample Data</h2>
            <table>
                <thead>
                    <tr>
"""

        # Add sample data
        for col in df.columns:
            html_content += f"<th>{col}</th>"

        html_content += """
                    </tr>
                </thead>
                <tbody>
"""

        for _, row in df.head().iterrows():
            html_content += "<tr>"
            for col in df.columns:
                value = str(row[col])[:50]  # Truncate long values
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"

        html_content += """
                </tbody>
            </table>
        </div>

        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Report generated by ML MCP System - HTML Report Generator</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"analysis_report_{timestamp}.html"

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML Report saved: {html_path}")
        return str(html_path)

    def _generate_enhanced_html_report(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> str:
        """Generate enhanced HTML report with detailed analysis"""

        if self.html_generator:
            # Use enhanced HTML generator
            html_content = self.html_generator.generate_comprehensive_report(
                df, self.analysis_results, dataset_name
            )
        else:
            # Fallback to basic HTML generation
            html_content = self._generate_basic_html_report(df, dataset_name)

        # Save HTML report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"enhanced_analysis_report_{timestamp}.html"

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Enhanced HTML Report saved: {html_path}")
        return str(html_path)

    def _generate_basic_html_report(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Generate basic HTML report as fallback"""

        from datetime import datetime

        # Basic HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basic Data Analysis Report - {dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; margin: -30px -30px 30px -30px; border-radius: 10px 10px 0 0; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Basic Data Analysis Report</h1>
            <h2>{dataset_name}</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by ML MCP System</p>
        </div>

        <div class="section">
            <h2>Dataset Overview</h2>
            <p><strong>Rows:</strong> {len(df):,}</p>
            <p><strong>Columns:</strong> {len(df.columns)}</p>
            <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
        </div>

        <div class="section">
            <h2>Sample Data</h2>
            <table>
                <thead>
                    <tr>{''.join([f'<th>{col}</th>' for col in df.columns])}</tr>
                </thead>
                <tbody>
                    {''.join([f'<tr>{"".join([f"<td>{str(val)[:50]}</td>" for val in row])}</tr>' for _, row in df.head().iterrows()])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Analysis Results</h2>
            <p>Detailed analysis results are available in the JSON file.</p>
        </div>

        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Report generated by ML MCP System</p>
        </div>
    </div>
</body>
</html>
"""
        return html_content

    def _save_json_results(self) -> str:
        """Save enhanced JSON results to file"""

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.json_organizer:
            # Use enhanced organized JSON structure
            json_path = self.output_dir / f"enhanced_analysis_results_{timestamp}.json"

            # Add insights and recommendations from analysis results
            all_insights = []
            all_recommendations = []

            for result in self.analysis_results:
                if "insights" in result:
                    all_insights.extend(result["insights"])
                if "recommendations" in result:
                    all_recommendations.extend(result["recommendations"])

            if all_insights:
                self.json_organizer.add_insights(all_insights)
            if all_recommendations:
                self.json_organizer.add_recommendations(all_recommendations)

            # Save organized results
            self.json_organizer.save_to_file(str(json_path))
            print(f"Enhanced JSON Results saved: {json_path}")
        else:
            # Fallback to basic JSON structure
            json_path = self.output_dir / f"analysis_results_{timestamp}.json"

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)

            print(f"JSON Results saved: {json_path}")

        return str(json_path)

    def _create_analysis_summary(self) -> Dict[str, Any]:
        """Create a summary of all analyses performed"""

        summary = {
            "total_analyses": len(self.analysis_results),
            "successful_analyses": len([r for r in self.analysis_results if r.get("success", False)]),
            "analysis_types": [r.get("analysis_type") for r in self.analysis_results if r.get("success", False)]
        }

        # Add key findings
        key_findings = []

        for result in self.analysis_results:
            if result.get("analysis_type") == "correlation_analysis":
                strong_corrs = result.get("strong_correlations", [])
                if strong_corrs:
                    key_findings.append(f"Found {len(strong_corrs)} significant correlations")

            elif result.get("analysis_type") == "missing_data_analysis":
                missing_pct = result.get("missing_data_summary", {}).get("missing_percentage", 0)
                if missing_pct > 0:
                    key_findings.append(f"Dataset has {missing_pct:.1f}% missing data")
                else:
                    key_findings.append("No missing data detected")

        summary["key_findings"] = key_findings
        return summary

def main():
    """Main function for CLI usage"""
    try:
        import argparse

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Integrated Data Analyzer")
        parser.add_argument("data_path", help="Path to the data file")
        parser.add_argument("output_dir", help="Output directory for results")
        args = parser.parse_args()

        # Load data from file
        data_path = Path(args.data_path)
        output_dir = args.output_dir

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load data based on file extension
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.json':
            df = pd.read_json(data_path)
        elif data_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            # Try CSV as default
            df = pd.read_csv(data_path)

        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")

        # Initialize integrated analyzer
        analyzer = IntegratedAnalyzer(output_dir)

        # Extract dataset name from path
        dataset_name = data_path.stem

        # Perform complete analysis
        result = analyzer.analyze_data(df, dataset_name)

        print("SUCCESS: Integrated analysis completed successfully!")
        print(f"Results saved in: {output_dir}")

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "integrated_analysis"
        }
        print(f"ERROR: Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()