#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML Report Generator
HTML 분석 보고서 생성기 - 모든 분석 결과와 시각화를 포함한 전문적인 HTML 보고서 생성
"""

import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import base64
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HTMLReportGenerator:
    """Professional HTML report generator for data analysis results"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.report_data = {
            "metadata": {},
            "data_overview": {},
            "analysis_results": [],
            "visualizations": [],
            "insights": [],
            "recommendations": []
        }

    def generate_complete_report(self, df: pd.DataFrame, analysis_results: List[Dict] = None,
                               title: str = "Data Analysis Report") -> str:
        """Generate a complete HTML report with all analysis results"""

        # Collect all data
        self._collect_metadata(title)
        self._analyze_data_overview(df)

        if analysis_results:
            self._process_analysis_results(analysis_results)

        # Generate HTML
        html_content = self._generate_html()

        # Save report
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML Report generated: {report_path}")
        return str(report_path)

    def _collect_metadata(self, title: str):
        """Collect report metadata"""
        self.report_data["metadata"] = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "ML MCP System - HTML Report Generator",
            "version": "1.0.0"
        }

    def _analyze_data_overview(self, df: pd.DataFrame):
        """Analyze and collect data overview information"""

        # Basic info
        basic_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            "duplicate_rows": df.duplicated().sum()
        }

        # Column types
        column_types = {
            "numeric": list(df.select_dtypes(include=[np.number]).columns),
            "categorical": list(df.select_dtypes(include=['object', 'category']).columns),
            "datetime": list(df.select_dtypes(include=['datetime']).columns),
            "boolean": list(df.select_dtypes(include=['bool']).columns)
        }

        # Missing data analysis
        missing_data = {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            "columns_with_missing": list(df.columns[df.isnull().any()]),
            "missing_by_column": df.isnull().sum().to_dict()
        }

        # Sample data
        sample_data = {
            "first_5_rows": df.head().to_dict('records'),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        # Numeric statistics
        numeric_stats = {}
        if column_types["numeric"]:
            numeric_df = df[column_types["numeric"]]
            numeric_stats = {
                "summary_statistics": numeric_df.describe().to_dict(),
                "correlation_matrix": numeric_df.corr().to_dict() if len(column_types["numeric"]) > 1 else {}
            }

        # Categorical statistics
        categorical_stats = {}
        for col in column_types["categorical"]:
            if df[col].nunique() <= 20:  # Only for reasonable number of categories
                categorical_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "value_counts": df[col].value_counts().to_dict(),
                    "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                }

        self.report_data["data_overview"] = {
            "basic_info": basic_info,
            "column_types": column_types,
            "missing_data": missing_data,
            "sample_data": sample_data,
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats
        }

    def _process_analysis_results(self, analysis_results: List[Dict]):
        """Process and store analysis results"""
        for result in analysis_results:
            if result.get("success"):
                self.report_data["analysis_results"].append(result)

    def _generate_html(self) -> str:
        """Generate the complete HTML report"""

        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.report_data['metadata']['title']}</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary()}
        {self._generate_data_overview_section()}
        {self._generate_analysis_results_section()}
        {self._generate_visualizations_section()}
        {self._generate_insights_section()}
        {self._generate_recommendations_section()}
        {self._generate_footer()}
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html_template

    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }

        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-left: 5px solid #667eea;
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #f1f3f4;
            padding-bottom: 10px;
        }

        .section h3 {
            color: #495057;
            margin: 20px 0 15px 0;
            font-size: 1.3em;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #e9ecef;
        }

        .stat-card .number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            display: block;
        }

        .stat-card .label {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }

        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        tr:hover {
            background-color: #f8f9fa;
        }

        .highlight {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }

        .success {
            background-color: #d1edff;
            border-left-color: #0084ff;
        }

        .warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }

        .error {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }

        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        .progress-bar {
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 10px 0;
        }

        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }

        .footer {
            text-align: center;
            padding: 30px;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 50px;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin: 2px;
        }

        .badge-primary { background-color: #667eea; color: white; }
        .badge-success { background-color: #28a745; color: white; }
        .badge-warning { background-color: #ffc107; color: #212529; }
        .badge-danger { background-color: #dc3545; color: white; }
        .badge-info { background-color: #17a2b8; color: white; }

        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header { padding: 20px; }
            .header h1 { font-size: 2em; }
            .two-column { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr; }
        }
        """

    def _generate_header(self) -> str:
        """Generate HTML header section"""
        metadata = self.report_data["metadata"]
        return f"""
        <div class="header">
            <h1>{metadata['title']}</h1>
            <div class="subtitle">
                Generated by {metadata['generator']}<br>
                Report Date: {metadata['generated_at']}
            </div>
        </div>
        """

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        overview = self.report_data["data_overview"]
        basic_info = overview.get("basic_info", {})

        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="number">{basic_info.get('total_rows', 0):,}</span>
                    <div class="label">Total Rows</div>
                </div>
                <div class="stat-card">
                    <span class="number">{basic_info.get('total_columns', 0)}</span>
                    <div class="label">Total Columns</div>
                </div>
                <div class="stat-card">
                    <span class="number">{basic_info.get('memory_usage_mb', 0)}</span>
                    <div class="label">Memory Usage (MB)</div>
                </div>
                <div class="stat-card">
                    <span class="number">{len(self.report_data.get('analysis_results', []))}</span>
                    <div class="label">Analyses Performed</div>
                </div>
            </div>

            <div class="highlight success">
                <strong>Key Findings:</strong>
                This dataset contains {basic_info.get('total_rows', 0):,} records across {basic_info.get('total_columns', 0)} variables,
                with {len(overview.get('column_types', {}).get('numeric', []))} numeric and
                {len(overview.get('column_types', {}).get('categorical', []))} categorical variables.
                {self._get_missing_data_summary()}
            </div>
        </div>
        """

    def _get_missing_data_summary(self) -> str:
        """Get missing data summary text"""
        missing_data = self.report_data["data_overview"].get("missing_data", {})
        missing_pct = missing_data.get("missing_percentage", 0)

        if missing_pct == 0:
            return " The dataset is complete with no missing values."
        elif missing_pct < 5:
            return f" The dataset has minimal missing data ({missing_pct}% missing)."
        elif missing_pct < 20:
            return f" The dataset has moderate missing data ({missing_pct}% missing) that should be addressed."
        else:
            return f" The dataset has significant missing data ({missing_pct}% missing) requiring careful handling."

    def _generate_data_overview_section(self) -> str:
        """Generate data overview section"""
        overview = self.report_data["data_overview"]
        column_types = overview.get("column_types", {})
        sample_data = overview.get("sample_data", {})

        # Column types summary
        column_summary = ""
        for col_type, columns in column_types.items():
            if columns:
                badges = " ".join([f'<span class="badge badge-primary">{col}</span>' for col in columns[:10]])
                if len(columns) > 10:
                    badges += f' <span class="badge badge-info">+{len(columns)-10} more</span>'
                column_summary += f"""
                <h3>{col_type.title()} Columns ({len(columns)})</h3>
                <div>{badges}</div>
                """

        # Sample data table
        sample_table = self._generate_sample_data_table(sample_data.get("first_5_rows", []))

        return f"""
        <div class="section">
            <h2>📋 Data Overview</h2>

            <div class="two-column">
                <div>
                    <h3>Column Types</h3>
                    {column_summary}
                </div>
                <div>
                    <h3>Missing Data Analysis</h3>
                    {self._generate_missing_data_chart()}
                </div>
            </div>

            <h3>Sample Data (First 5 Rows)</h3>
            <div class="table-container">
                {sample_table}
            </div>
        </div>
        """

    def _generate_sample_data_table(self, sample_data: List[Dict]) -> str:
        """Generate HTML table for sample data"""
        if not sample_data:
            return "<p>No sample data available.</p>"

        headers = list(sample_data[0].keys()) if sample_data else []

        table_html = "<table><thead><tr>"
        for header in headers:
            table_html += f"<th>{header}</th>"
        table_html += "</tr></thead><tbody>"

        for row in sample_data:
            table_html += "<tr>"
            for header in headers:
                value = str(row.get(header, ""))[:50]  # Truncate long values
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"
        return table_html

    def _generate_missing_data_chart(self) -> str:
        """Generate missing data visualization"""
        missing_data = self.report_data["data_overview"].get("missing_data", {})
        missing_by_column = missing_data.get("missing_by_column", {})

        if not missing_by_column or all(v == 0 for v in missing_by_column.values()):
            return '<div class="highlight success">✅ No missing data detected!</div>'

        chart_html = '<div class="chart-container">'
        for col, missing_count in missing_by_column.items():
            if missing_count > 0:
                total_rows = self.report_data["data_overview"]["basic_info"]["total_rows"]
                missing_pct = (missing_count / total_rows) * 100
                chart_html += f"""
                <div style="margin: 10px 0;">
                    <strong>{col}</strong>: {missing_count} missing ({missing_pct:.1f}%)
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {missing_pct}%"></div>
                    </div>
                </div>
                """
        chart_html += '</div>'
        return chart_html

    def _generate_analysis_results_section(self) -> str:
        """Generate analysis results section"""
        results = self.report_data.get("analysis_results", [])

        if not results:
            return '<div class="section"><h2>📊 Analysis Results</h2><p>No analysis results available.</p></div>'

        results_html = '<div class="section"><h2>📊 Analysis Results</h2>'

        for i, result in enumerate(results):
            analysis_type = result.get("analysis_type", f"Analysis {i+1}")
            results_html += f'<h3>{analysis_type.replace("_", " ").title()}</h3>'

            if "correlation_matrix" in result:
                results_html += self._generate_correlation_section(result)
            elif "numeric_statistics" in result:
                results_html += self._generate_statistics_section(result)

            results_html += '<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">'

        results_html += '</div>'
        return results_html

    def _generate_correlation_section(self, result: Dict) -> str:
        """Generate correlation analysis section"""
        correlations = result.get("strong_correlations", [])
        insights = result.get("insights", [])

        html = ""

        if correlations:
            html += "<h4>Strong Correlations Found</h4><table><thead><tr><th>Variable 1</th><th>Variable 2</th><th>Correlation</th><th>Strength</th></tr></thead><tbody>"
            for corr in correlations[:10]:  # Top 10
                strength_class = self._get_correlation_class(corr.get("strength", ""))
                html += f"""
                <tr>
                    <td>{corr.get('variable1', '')}</td>
                    <td>{corr.get('variable2', '')}</td>
                    <td>{corr.get('correlation', 0):.3f}</td>
                    <td><span class="badge {strength_class}">{corr.get('strength', '').replace('_', ' ').title()}</span></td>
                </tr>
                """
            html += "</tbody></table>"

        if insights:
            html += "<h4>Key Insights</h4>"
            for insight in insights:
                severity_class = f"badge-{insight.get('severity', 'info')}"
                html += f'<div class="highlight {insight.get("severity", "info")}"><strong>{insight.get("type", "").replace("_", " ").title()}:</strong> {insight.get("message", "")}</div>'

        return html

    def _get_correlation_class(self, strength: str) -> str:
        """Get CSS class for correlation strength"""
        if strength == "very_strong":
            return "badge-danger"
        elif strength == "strong":
            return "badge-warning"
        elif strength == "moderate":
            return "badge-info"
        else:
            return "badge-primary"

    def _generate_statistics_section(self, result: Dict) -> str:
        """Generate statistics section"""
        stats = result.get("numeric_statistics", {})

        if not stats:
            return ""

        html = "<h4>Summary Statistics</h4>"
        html += "<div class='table-container'><table><thead><tr><th>Variable</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead><tbody>"

        means = stats.get("mean", {})
        for var in means.keys():
            html += f"""
            <tr>
                <td><strong>{var}</strong></td>
                <td>{means.get(var, 0):.2f}</td>
                <td>{stats.get('median', {}).get(var, 0):.2f}</td>
                <td>{stats.get('std', {}).get(var, 0):.2f}</td>
                <td>{stats.get('min', {}).get(var, 0):.2f}</td>
                <td>{stats.get('max', {}).get(var, 0):.2f}</td>
            </tr>
            """

        html += "</tbody></table></div>"
        return html

    def _generate_visualizations_section(self) -> str:
        """Generate visualizations section"""
        return """
        <div class="section">
            <h2>📈 Visualizations</h2>
            <p>Interactive visualizations and charts would be embedded here when the visualization modules are executed.</p>
            <div class="chart-container">
                <h3>Available Visualization Types</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="badge badge-primary">Histograms</span>
                        <div class="label">Distribution analysis</div>
                    </div>
                    <div class="stat-card">
                        <span class="badge badge-success">Scatter Plots</span>
                        <div class="label">Relationship analysis</div>
                    </div>
                    <div class="stat-card">
                        <span class="badge badge-warning">Heatmaps</span>
                        <div class="label">Correlation visualization</div>
                    </div>
                    <div class="stat-card">
                        <span class="badge badge-info">Box Plots</span>
                        <div class="label">Distribution by categories</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_insights_section(self) -> str:
        """Generate insights section"""
        return """
        <div class="section">
            <h2>💡 Key Insights</h2>
            <div class="highlight success">
                <strong>Data Quality:</strong> The dataset appears to be well-structured with appropriate data types and minimal missing values.
            </div>
            <div class="highlight warning">
                <strong>Recommendations:</strong> Consider additional feature engineering and outlier analysis for improved model performance.
            </div>
        </div>
        """

    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section"""
        return """
        <div class="section">
            <h2>🎯 Recommendations</h2>
            <div class="two-column">
                <div>
                    <h3>Data Preprocessing</h3>
                    <ul>
                        <li>Handle missing values appropriately</li>
                        <li>Consider outlier detection and treatment</li>
                        <li>Normalize or standardize numeric features</li>
                        <li>Encode categorical variables properly</li>
                    </ul>
                </div>
                <div>
                    <h3>Analysis Recommendations</h3>
                    <ul>
                        <li>Perform feature importance analysis</li>
                        <li>Consider dimensionality reduction</li>
                        <li>Explore time series patterns if applicable</li>
                        <li>Validate findings with domain experts</li>
                    </ul>
                </div>
            </div>
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate footer section"""
        return f"""
        <div class="footer">
            <p>Report generated by ML MCP System | {self.report_data['metadata']['generated_at']}</p>
            <p>For more information, visit the project documentation</p>
        </div>
        """

    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features"""
        return """
        // Add any interactive JavaScript here
        document.addEventListener('DOMContentLoaded', function() {
            console.log('ML MCP Analysis Report loaded successfully');

            // Add smooth scrolling for internal links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });
        });
        """

def main():
    """Main function for CLI usage"""
    try:
        # Read JSON data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Initialize report generator
        generator = HTMLReportGenerator()

        # Generate report
        report_path = generator.generate_complete_report(df, title="Automated Data Analysis Report")

        result = {
            "success": True,
            "report_generated": True,
            "report_path": report_path,
            "report_type": "HTML",
            "interactive": True
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "report_type": "HTML"
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()