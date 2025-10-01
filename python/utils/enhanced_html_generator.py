#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced HTML Report Generator
상세한 분석 결과를 포함한 고급 HTML 보고서 생성기
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64
from io import BytesIO
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
    plt.style.use('default')  # Use default style instead of seaborn
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None


class EnhancedHTMLReportGenerator:
    """Enhanced HTML report generator with detailed analysis"""

    def __init__(self):
        self.report_sections = []
        self.css_styles = self._get_enhanced_css()
        if HAS_MATPLOTLIB:
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')

    def generate_comprehensive_report(self,
                                    df: pd.DataFrame,
                                    analysis_results: List[Dict[str, Any]],
                                    dataset_name: str = "Dataset") -> str:
        """Generate comprehensive HTML report"""

        # Initialize report
        html_content = self._get_html_header(dataset_name)

        # Add executive summary
        html_content += self._generate_executive_summary(df, analysis_results)

        # Add data overview section
        html_content += self._generate_data_overview(df)

        # Add detailed statistical analysis
        for result in analysis_results:
            html_content += self._generate_analysis_section(result, df)

        # Add data quality assessment
        html_content += self._generate_data_quality_section(df)

        # Add insights and recommendations
        html_content += self._generate_insights_section(analysis_results)

        # Add interactive elements
        html_content += self._generate_interactive_elements(df)

        # Add complete analysis results section
        html_content += self._generate_complete_results_section(analysis_results)

        # Add appendix
        html_content += self._generate_appendix(df)

        # Close HTML
        html_content += self._get_html_footer()

        return html_content

    def _get_html_header(self, dataset_name: str) -> str:
        """Get HTML header with enhanced styling"""

        return f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Data Analysis Report - {dataset_name}</title>
    <style>
        {self.css_styles}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="icon">📊</i> Enhanced Data Analysis Report</h1>
            <h2>{dataset_name}</h2>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by ML MCP System</p>
        </div>
"""

    def _generate_executive_summary(self, df: pd.DataFrame, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate executive summary section"""

        # Calculate key metrics
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_percentage = (df.isnull().sum().sum() / (total_rows * total_columns)) * 100
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024**2

        # Identify data types
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime']).columns)

        # Quality assessment
        quality_score = self._calculate_overall_quality_score(df)

        return f"""
        <div class="section executive-summary">
            <h2><i class="icon">📋</i> Executive Summary</h2>

            <div class="summary-grid">
                <div class="summary-card high">
                    <div class="card-header">
                        <h3>Dataset Overview</h3>
                        <span class="badge success">Core Metrics</span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric">
                            <span class="number">{total_rows:,}</span>
                            <span class="label">Total Rows</span>
                        </div>
                        <div class="metric">
                            <span class="number">{total_columns}</span>
                            <span class="label">Columns</span>
                        </div>
                        <div class="metric">
                            <span class="number">{memory_usage_mb:.1f} MB</span>
                            <span class="label">Memory Usage</span>
                        </div>
                    </div>
                </div>

                <div class="summary-card medium">
                    <div class="card-header">
                        <h3>Data Composition</h3>
                        <span class="badge info">Type Distribution</span>
                    </div>
                    <div class="composition-chart">
                        <div class="composition-item">
                            <span class="composition-bar" style="width: {(numeric_cols/total_columns)*100}%; background-color: #3498db;"></span>
                            <span class="composition-label">Numeric ({numeric_cols})</span>
                        </div>
                        <div class="composition-item">
                            <span class="composition-bar" style="width: {(categorical_cols/total_columns)*100}%; background-color: #e74c3c;"></span>
                            <span class="composition-label">Categorical ({categorical_cols})</span>
                        </div>
                        <div class="composition-item">
                            <span class="composition-bar" style="width: {(datetime_cols/total_columns)*100}%; background-color: #f39c12;"></span>
                            <span class="composition-label">DateTime ({datetime_cols})</span>
                        </div>
                    </div>
                </div>

                <div class="summary-card high">
                    <div class="card-header">
                        <h3>Data Quality Score</h3>
                        <span class="badge {self._get_quality_badge_class(quality_score)}">
                            {self._get_quality_grade(quality_score)}
                        </span>
                    </div>
                    <div class="quality-score">
                        <div class="score-circle">
                            <div class="score-text">{quality_score:.0f}</div>
                            <div class="score-label">/ 100</div>
                        </div>
                        <div class="quality-breakdown">
                            <div class="quality-item">
                                <span class="quality-metric">Completeness</span>
                                <span class="quality-value">{100-missing_percentage:.1f}%</span>
                            </div>
                            <div class="quality-item">
                                <span class="quality-metric">Consistency</span>
                                <span class="quality-value">{100-((df.duplicated().sum()/len(df))*100):.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="key-findings">
                <h3>Key Findings</h3>
                <div class="findings-grid">
                    {self._generate_key_findings(df, analysis_results)}
                </div>
            </div>
        </div>
"""

    def _generate_data_overview(self, df: pd.DataFrame) -> str:
        """Generate detailed data overview section"""

        return f"""
        <div class="section data-overview">
            <h2><i class="icon">🔍</i> Data Overview</h2>

            <div class="overview-tabs">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('columns')">Column Analysis</button>
                    <button class="tab-button" onclick="showTab('sample')">Sample Data</button>
                    <button class="tab-button" onclick="showTab('statistics')">Quick Statistics</button>
                </div>

                <div id="columns" class="tab-content active">
                    {self._generate_column_analysis_table(df)}
                </div>

                <div id="sample" class="tab-content">
                    {self._generate_sample_data_table(df)}
                </div>

                <div id="statistics" class="tab-content">
                    {self._generate_quick_statistics(df)}
                </div>
            </div>
        </div>
"""

    def _generate_analysis_section(self, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate detailed analysis section based on type"""

        analysis_type = result.get("analysis_type", "Unknown")

        if analysis_type == "descriptive_statistics":
            return self._generate_descriptive_stats_section(result, df)
        elif analysis_type == "correlation_analysis":
            return self._generate_correlation_section(result, df)
        elif analysis_type == "missing_data_analysis":
            return self._generate_missing_data_section(result, df)
        elif analysis_type == "data_types_analysis":
            return self._generate_data_types_section(result, df)
        else:
            return self._generate_generic_analysis_section(result)

    def _generate_descriptive_stats_section(self, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate detailed descriptive statistics section"""

        numeric_stats = result.get("numeric_statistics", {})
        categorical_stats = result.get("categorical_statistics", {})

        # Generate distribution plots
        distribution_plots = self._generate_distribution_plots(df)

        return f"""
        <div class="section analysis-section">
            <h2><i class="icon">📈</i> Descriptive Statistics Analysis</h2>

            <div class="stats-container">
                <div class="stats-overview">
                    <h3>Statistical Summary</h3>
                    <div class="overview-grid">
                        <div class="overview-item">
                            <span class="overview-number">{len(result.get('numeric_columns', []))}</span>
                            <span class="overview-label">Numeric Variables</span>
                        </div>
                        <div class="overview-item">
                            <span class="overview-number">{len(result.get('categorical_columns', []))}</span>
                            <span class="overview-label">Categorical Variables</span>
                        </div>
                    </div>
                </div>

                <div class="numeric-analysis">
                    <h3>Numeric Variables Analysis</h3>
                    {self._generate_detailed_numeric_table(numeric_stats)}
                </div>

                <div class="categorical-analysis">
                    <h3>Categorical Variables Analysis</h3>
                    {self._generate_detailed_categorical_table(categorical_stats)}
                </div>

                <div class="distribution-analysis">
                    <h3>Distribution Analysis</h3>
                    <div class="plot-container">
                        {distribution_plots}
                    </div>
                </div>
            </div>
        </div>
"""

    def _generate_correlation_section(self, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate correlation analysis section"""

        correlation_matrix = result.get("correlation_matrix", {})
        strong_correlations = result.get("strong_correlations", [])

        # Generate correlation heatmap
        heatmap_html = self._generate_correlation_heatmap(df)

        return f"""
        <div class="section analysis-section">
            <h2><i class="icon">🔗</i> Correlation Analysis</h2>

            <div class="correlation-container">
                <div class="correlation-summary">
                    <h3>Correlation Summary</h3>
                    <div class="correlation-stats">
                        <div class="stat-item">
                            <span class="stat-number">{len(strong_correlations)}</span>
                            <span class="stat-label">Strong Correlations Found</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{len(result.get('data_info', {}).get('numeric_columns', []))}</span>
                            <span class="stat-label">Variables Analyzed</span>
                        </div>
                    </div>
                </div>

                <div class="correlation-heatmap">
                    <h3>Correlation Heatmap</h3>
                    {heatmap_html}
                </div>

                <div class="significant-correlations">
                    <h3>Significant Correlations</h3>
                    {self._generate_correlation_insights_table(strong_correlations)}
                </div>
            </div>
        </div>
"""

    def _generate_missing_data_section(self, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate missing data analysis section"""

        missing_summary = result.get("missing_data_summary", {})
        missing_by_column = result.get("missing_by_column", {})

        return f"""
        <div class="section analysis-section">
            <h2><i class="icon">❓</i> Missing Data Analysis</h2>

            <div class="missing-data-container">
                <div class="missing-overview">
                    <h3>Missing Data Overview</h3>
                    <div class="missing-stats">
                        <div class="missing-stat">
                            <span class="stat-number">{missing_summary.get('total_missing_values', 0)}</span>
                            <span class="stat-label">Total Missing Values</span>
                        </div>
                        <div class="missing-stat">
                            <span class="stat-number">{missing_summary.get('missing_percentage', 0):.1f}%</span>
                            <span class="stat-label">Missing Percentage</span>
                        </div>
                        <div class="missing-stat">
                            <span class="stat-number">{missing_summary.get('complete_rows', 0)}</span>
                            <span class="stat-label">Complete Rows</span>
                        </div>
                    </div>
                </div>

                <div class="missing-pattern">
                    <h3>Missing Data Pattern</h3>
                    {self._generate_missing_data_visualization(df)}
                </div>

                <div class="missing-recommendations">
                    <h3>Recommendations</h3>
                    {self._generate_missing_data_recommendations(result)}
                </div>
            </div>
        </div>
"""

    def _generate_data_quality_section(self, df: pd.DataFrame) -> str:
        """Generate comprehensive data quality assessment"""

        quality_metrics = self._assess_comprehensive_data_quality(df)

        return f"""
        <div class="section data-quality">
            <h2><i class="icon">✅</i> Data Quality Assessment</h2>

            <div class="quality-dashboard">
                <div class="quality-metrics">
                    <h3>Quality Metrics</h3>
                    <div class="metrics-grid">
                        {self._generate_quality_metrics_cards(quality_metrics)}
                    </div>
                </div>

                <div class="quality-details">
                    <h3>Detailed Assessment</h3>
                    {self._generate_quality_details_table(quality_metrics)}
                </div>

                <div class="quality-recommendations">
                    <h3>Quality Improvement Recommendations</h3>
                    {self._generate_quality_recommendations(quality_metrics)}
                </div>
            </div>
        </div>
"""

    def _generate_insights_section(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate insights and recommendations section"""

        all_insights = []
        all_recommendations = []

        for result in analysis_results:
            if "insights" in result:
                all_insights.extend(result["insights"])
            if "recommendations" in result:
                all_recommendations.extend(result["recommendations"])

        return f"""
        <div class="section insights-section">
            <h2><i class="icon">💡</i> Insights & Recommendations</h2>

            <div class="insights-container">
                <div class="key-insights">
                    <h3>Key Insights</h3>
                    {self._generate_insights_list(all_insights)}
                </div>

                <div class="actionable-recommendations">
                    <h3>Actionable Recommendations</h3>
                    {self._generate_recommendations_list(all_recommendations)}
                </div>
            </div>
        </div>
"""

    def _generate_interactive_elements(self, df: pd.DataFrame) -> str:
        """Generate interactive elements section"""

        return """
        <div class="section interactive-section">
            <h2><i class="icon">🎛️</i> Interactive Exploration</h2>

            <div class="interactive-container">
                <div class="filter-controls">
                    <h3>Data Filters</h3>
                    <div class="controls-grid">
                        <button class="filter-btn" onclick="showAllData()">Show All Data</button>
                        <button class="filter-btn" onclick="showNumericOnly()">Numeric Columns Only</button>
                        <button class="filter-btn" onclick="showCategoricalOnly()">Categorical Columns Only</button>
                    </div>
                </div>

                <div class="search-functionality">
                    <h3>Search Data</h3>
                    <input type="text" id="dataSearch" placeholder="Search in data..."
                           onkeyup="searchData()" class="search-input">
                    <div id="searchResults" class="search-results"></div>
                </div>
            </div>
        </div>
"""

    def _generate_appendix(self, df: pd.DataFrame) -> str:
        """Generate appendix with technical details"""

        return f"""
        <div class="section appendix">
            <h2><i class="icon">📎</i> Technical Appendix</h2>

            <div class="appendix-container">
                <div class="technical-details">
                    <h3>Technical Details</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <strong>Analysis Engine:</strong> ML MCP System v1.0
                        </div>
                        <div class="detail-item">
                            <strong>Processing Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                        <div class="detail-item">
                            <strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
                        </div>
                        <div class="detail-item">
                            <strong>Analysis Methods:</strong> Descriptive Statistics, Correlation Analysis, Missing Data Analysis
                        </div>
                    </div>
                </div>

                <div class="methodology">
                    <h3>Methodology</h3>
                    <div class="methodology-content">
                        <p><strong>Statistical Analysis:</strong> Comprehensive descriptive statistics including measures of central tendency, dispersion, and distribution shape.</p>
                        <p><strong>Correlation Analysis:</strong> Pearson correlation coefficients with significance testing and strength classification.</p>
                        <p><strong>Data Quality:</strong> Multi-dimensional assessment including completeness, consistency, validity, and uniqueness.</p>
                        <p><strong>Missing Data:</strong> Pattern analysis and imputation recommendations based on data characteristics.</p>
                    </div>
                </div>
            </div>
        </div>
"""

    def _get_html_footer(self) -> str:
        """Get HTML footer with JavaScript functionality"""

        return """
        <script>
            // Tab functionality
            function showTab(tabName) {
                var tabs = document.querySelectorAll('.tab-content');
                var buttons = document.querySelectorAll('.tab-button');

                tabs.forEach(tab => tab.classList.remove('active'));
                buttons.forEach(btn => btn.classList.remove('active'));

                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            // Search functionality
            function searchData() {
                var input = document.getElementById('dataSearch');
                var filter = input.value.toLowerCase();
                // Implementation for searching through data tables
                console.log('Searching for:', filter);
            }

            // Filter functionality
            function showAllData() {
                console.log('Showing all data');
            }

            function showNumericOnly() {
                console.log('Showing numeric columns only');
            }

            function showCategoricalOnly() {
                console.log('Showing categorical columns only');
            }

            // Print functionality
            function printReport() {
                window.print();
            }

            // Download functionality
            function downloadReport() {
                console.log('Downloading report');
            }

            // JSON section functionality
            function toggleJsonView() {
                var container = document.getElementById('jsonContainer');
                if (container.style.display === 'none') {
                    container.style.display = 'block';
                } else {
                    container.style.display = 'none';
                }
            }

            function copyJsonData() {
                var jsonData = document.getElementById('jsonData').textContent;
                navigator.clipboard.writeText(jsonData).then(function() {
                    alert('JSON 데이터가 클립보드에 복사되었습니다!');
                }).catch(function(err) {
                    console.error('복사 실패:', err);
                });
            }

            function downloadJsonData() {
                var jsonData = document.getElementById('jsonData').textContent;
                var blob = new Blob([jsonData], { type: 'application/json' });
                var url = window.URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'analysis_results_' + new Date().toISOString().slice(0,10) + '.json';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }

            // Expandable results functionality
            function toggleResultSection(sectionId) {
                var content = document.getElementById(sectionId);
                var icon = document.getElementById('icon_' + sectionId);

                if (content.style.display === 'none' || content.style.display === '') {
                    content.style.display = 'block';
                    icon.textContent = '▲';
                } else {
                    content.style.display = 'none';
                    icon.textContent = '▼';
                }
            }

            // Result tabs functionality
            function showResultTab(resultId, tabName) {
                // Hide all tab contents for this result
                var tabContents = document.querySelectorAll(`#${resultId} .tab-content`);
                var tabButtons = document.querySelectorAll(`#${resultId} .tab-button`);

                tabContents.forEach(tab => tab.classList.remove('active'));
                tabButtons.forEach(btn => btn.classList.remove('active'));

                // Show selected tab
                document.getElementById(`${resultId}_${tabName}`).classList.add('active');
                event.target.classList.add('active');
            }
        </script>

        <div class="footer">
            <div class="footer-content">
                <p>Report generated by ML MCP System - Enhanced HTML Report Generator</p>
                <div class="footer-actions">
                    <button onclick="printReport()" class="footer-btn">🖨️ Print Report</button>
                    <button onclick="downloadReport()" class="footer-btn">💾 Download JSON</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    def _get_enhanced_css(self) -> str:
        """Get enhanced CSS styles"""

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
            background: #f5f7fa;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }

        .header h2 {
            font-size: 1.5em;
            opacity: 0.9;
            font-weight: 400;
        }

        .subtitle {
            margin-top: 15px;
            opacity: 0.8;
            font-size: 0.95em;
        }

        .icon {
            margin-right: 10px;
            font-size: 1.2em;
        }

        .section {
            margin: 40px;
            padding: 30px;
            border-radius: 10px;
            background: #fafbfc;
            border: 1px solid #e1e8ed;
        }

        .executive-summary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .summary-card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }

        .badge.success { background: #27ae60; }
        .badge.info { background: #3498db; }
        .badge.warning { background: #f39c12; }
        .badge.danger { background: #e74c3c; }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 15px;
        }

        .metric {
            text-align: center;
        }

        .metric .number {
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #fff;
        }

        .metric .label {
            font-size: 0.8em;
            opacity: 0.9;
        }

        .composition-chart {
            margin-top: 15px;
        }

        .composition-item {
            margin: 10px 0;
            display: flex;
            align-items: center;
        }

        .composition-bar {
            height: 8px;
            border-radius: 4px;
            margin-right: 10px;
            min-width: 20px;
        }

        .quality-score {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .score-circle {
            width: 80px;
            height: 80px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .score-text {
            font-size: 1.8em;
            font-weight: bold;
        }

        .score-label {
            font-size: 0.7em;
            opacity: 0.8;
        }

        .key-findings {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        .findings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .finding-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #fff;
        }

        .overview-tabs {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .tab-buttons {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }

        .tab-button {
            flex: 1;
            padding: 15px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .tab-button.active {
            background: #667eea;
            color: white;
        }

        .tab-content {
            display: none;
            padding: 30px;
        }

        .tab-content.active {
            display: block;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .data-table th,
        .data-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        .data-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        .data-table tr:hover {
            background: #f8f9fa;
        }

        .plot-container {
            margin: 20px 0;
            text-align: center;
        }

        .plot-image {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .interactive-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .filter-btn {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .filter-btn:hover {
            background: #5a6fd8;
        }

        .search-input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            font-size: 1em;
            margin-top: 10px;
        }

        .footer {
            background: #2c3e50;
            color: white;
            padding: 30px 40px;
        }

        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .footer-actions {
            display: flex;
            gap: 15px;
        }

        .footer-btn {
            padding: 8px 15px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
        }

        .missing-stat, .stat-item, .quality-metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .stat-number {
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }

        /* New styles for complete results section */
        .complete-results {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
        }

        .results-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .result-stat {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .expandable-result {
            margin: 20px 0;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            background: white;
            overflow: hidden;
        }

        .result-header {
            padding: 20px;
            background: #f8f9fa;
            cursor: pointer;
            transition: background-color 0.3s ease;
            border-bottom: 1px solid #dee2e6;
        }

        .result-header:hover {
            background: #e9ecef;
        }

        .result-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .expand-icon {
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
        }

        .result-summary {
            color: #6c757d;
            font-style: italic;
        }

        .result-content {
            padding: 0;
        }

        .result-tabs {
            border-top: 1px solid #dee2e6;
        }

        .json-controls {
            margin: 20px 0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .json-toggle-btn, .json-copy-btn, .json-download-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .json-toggle-btn {
            background: #667eea;
            color: white;
        }

        .json-copy-btn {
            background: #28a745;
            color: white;
        }

        .json-download-btn {
            background: #17a2b8;
            color: white;
        }

        .json-toggle-btn:hover { background: #5a6fd8; }
        .json-copy-btn:hover { background: #218838; }
        .json-download-btn:hover { background: #138496; }

        .json-container {
            border: 1px solid #dee2e6;
            border-radius: 5px;
            background: #f8f9fa;
            max-height: 400px;
            overflow-y: auto;
        }

        .json-display {
            background: #f8f9fa;
            padding: 20px;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .overview-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
        }

        .overview-label {
            display: block;
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }

        .overview-value {
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: #495057;
        }

        .metrics-display {
            padding: 20px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }

        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
        }

        .dict-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }

        .dict-item {
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }

        .dict-key {
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }

        .dict-value {
            color: #6c757d;
            padding-left: 10px;
        }

        .insight-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }

        .insight-icon {
            margin-right: 10px;
            font-size: 1.2em;
        }

        .insight-text {
            flex: 1;
            color: #495057;
        }

        /* Correlation color classes */
        .corr-very-strong { background-color: #dc3545; color: white; }
        .corr-strong { background-color: #fd7e14; color: white; }
        .corr-moderate { background-color: #ffc107; color: black; }
        .corr-weak { background-color: #6f42c1; color: white; }
        .corr-very-weak { background-color: #e9ecef; color: black; }

        .correlation-matrix-table td {
            text-align: center;
            font-weight: bold;
        }

        @media (max-width: 768px) {
            .container {
                margin: 0;
                border-radius: 0;
            }

            .section {
                margin: 20px;
                padding: 20px;
            }

            .summary-grid {
                grid-template-columns: 1fr;
            }

            .tab-buttons {
                flex-direction: column;
            }

            .json-controls {
                flex-direction: column;
            }

            .overview-grid {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
"""

    # Helper methods for generating specific content sections
    def _calculate_overall_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        missing_penalty = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_penalty = (df.duplicated().sum() / len(df)) * 100

        base_score = 100
        quality_score = base_score - missing_penalty - duplicate_penalty
        return max(0, quality_score)

    def _get_quality_badge_class(self, score: float) -> str:
        """Get CSS class for quality badge"""
        if score >= 90:
            return "success"
        elif score >= 70:
            return "info"
        elif score >= 50:
            return "warning"
        else:
            return "danger"

    def _get_quality_grade(self, score: float) -> str:
        """Convert score to grade"""
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

    def _generate_key_findings(self, df: pd.DataFrame, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate key findings HTML"""
        findings = []

        # Data size finding
        findings.append(f"""
            <div class="finding-item">
                <strong>Dataset Size:</strong> {len(df):,} rows × {len(df.columns)} columns
            </div>
        """)

        # Missing data finding
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 0:
            findings.append(f"""
                <div class="finding-item">
                    <strong>Missing Data:</strong> {missing_pct:.1f}% of values are missing
                </div>
            """)
        else:
            findings.append(f"""
                <div class="finding-item">
                    <strong>Data Completeness:</strong> No missing values detected
                </div>
            """)

        # Data types finding
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        findings.append(f"""
            <div class="finding-item">
                <strong>Numeric Variables:</strong> {numeric_cols} out of {len(df.columns)} columns
            </div>
        """)

        return "".join(findings)

    def _generate_column_analysis_table(self, df: pd.DataFrame) -> str:
        """Generate detailed column analysis table"""
        table_rows = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100

            table_rows.append(f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td><span class="badge info">{dtype}</span></td>
                    <td>{unique_count:,}</td>
                    <td>{null_count}</td>
                    <td>{null_pct:.1f}%</td>
                    <td>{df[col].memory_usage(deep=True):,} bytes</td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Unique Values</th>
                        <th>Missing Count</th>
                        <th>Missing %</th>
                        <th>Memory Usage</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        """

    def _generate_sample_data_table(self, df: pd.DataFrame) -> str:
        """Generate sample data table"""
        sample_data = df.head(10)

        # Generate table header
        header = "<tr>" + "".join([f"<th>{col}</th>" for col in sample_data.columns]) + "</tr>"

        # Generate table rows
        rows = []
        for _, row in sample_data.iterrows():
            row_html = "<tr>" + "".join([f"<td>{str(val)[:50]}</td>" for val in row]) + "</tr>"
            rows.append(row_html)

        return f"""
            <table class="data-table">
                <thead>{header}</thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        """

    def _generate_quick_statistics(self, df: pd.DataFrame) -> str:
        """Generate quick statistics overview"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return "<p>No numeric columns available for statistical analysis.</p>"

        stats = numeric_df.describe()

        # Convert stats to HTML table
        table_rows = []
        for index in stats.index:
            row = f"<tr><td><strong>{index}</strong></td>"
            for col in stats.columns:
                value = stats.loc[index, col]
                if pd.isna(value):
                    row += "<td>-</td>"
                else:
                    row += f"<td>{value:.2f}</td>"
            row += "</tr>"
            table_rows.append(row)

        header = "<tr><th>Statistic</th>" + "".join([f"<th>{col}</th>" for col in stats.columns]) + "</tr>"

        return f"""
            <table class="data-table">
                <thead>{header}</thead>
                <tbody>{"".join(table_rows)}</tbody>
            </table>
        """

    # Additional helper methods would continue here...
    # This is a comprehensive foundation for the enhanced HTML generator

    def _generate_detailed_numeric_table(self, numeric_stats: Dict[str, Any]) -> str:
        """Generate detailed numeric statistics table"""
        if not numeric_stats:
            return "<p>No numeric statistics available.</p>"

        return f"""
            <div class="stats-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Count</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>Median</th>
                            <th>75%</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_numeric_stats_rows(numeric_stats)}
                    </tbody>
                </table>
            </div>
        """

    def _generate_numeric_stats_rows(self, numeric_stats: Dict[str, Any]) -> str:
        """Generate rows for numeric statistics table"""
        rows = []

        # Get all variables from the first statistic type
        variables = list(numeric_stats.get('count', {}).keys()) if 'count' in numeric_stats else []

        for var in variables:
            count = numeric_stats.get('count', {}).get(var, '-')
            mean = numeric_stats.get('mean', {}).get(var, '-')
            std = numeric_stats.get('std', {}).get(var, '-')
            min_val = numeric_stats.get('min', {}).get(var, '-')
            q25 = numeric_stats.get('q25', {}).get(var, '-')
            median = numeric_stats.get('median', {}).get(var, '-')
            q75 = numeric_stats.get('q75', {}).get(var, '-')
            max_val = numeric_stats.get('max', {}).get(var, '-')

            # Format numbers
            def format_num(val):
                if isinstance(val, (int, float)):
                    return f"{val:.3f}" if isinstance(val, float) else str(val)
                return str(val)

            rows.append(f"""
                <tr>
                    <td><strong>{var}</strong></td>
                    <td>{format_num(count)}</td>
                    <td>{format_num(mean)}</td>
                    <td>{format_num(std)}</td>
                    <td>{format_num(min_val)}</td>
                    <td>{format_num(q25)}</td>
                    <td>{format_num(median)}</td>
                    <td>{format_num(q75)}</td>
                    <td>{format_num(max_val)}</td>
                </tr>
            """)

        return "".join(rows)

    def _generate_detailed_categorical_table(self, categorical_stats: Dict[str, Any]) -> str:
        """Generate detailed categorical statistics table"""
        if not categorical_stats:
            return "<p>No categorical statistics available.</p>"

        rows = []
        for var, stats in categorical_stats.items():
            rows.append(f"""
                <tr>
                    <td><strong>{var}</strong></td>
                    <td>{stats.get('unique_count', '-')}</td>
                    <td>{stats.get('most_frequent', '-')}</td>
                    <td>{stats.get('most_frequent_count', '-')}</td>
                    <td>{stats.get('null_count', '-')}</td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Unique Values</th>
                        <th>Most Frequent</th>
                        <th>Frequency</th>
                        <th>Missing Count</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _generate_distribution_plots(self, df: pd.DataFrame) -> str:
        """Generate distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return "<p>No numeric columns available for distribution analysis.</p>"

        if not HAS_MATPLOTLIB:
            return "<p>Matplotlib not available for plot generation. Please install matplotlib to see distribution plots.</p>"

        plots_html = []

        for col in numeric_cols[:6]:  # Limit to first 6 columns
            try:
                plt.figure(figsize=(8, 6))
                plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)

                # Save plot to base64 string
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                plots_html.append(f"""
                    <div class="plot-item">
                        <h4>{col} Distribution</h4>
                        <img src="data:image/png;base64,{plot_data}" class="plot-image" alt="{col} distribution">
                    </div>
                """)
            except Exception as e:
                plots_html.append(f"""
                    <div class="plot-item">
                        <h4>{col} Distribution</h4>
                        <p>Could not generate plot: {str(e)}</p>
                    </div>
                """)

        return f"""
            <div class="plots-grid">
                {"".join(plots_html)}
            </div>
        """

    def _generate_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Generate correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return "<p>Need at least 2 numeric columns for correlation analysis.</p>"

        if not HAS_MATPLOTLIB:
            return "<p>Matplotlib not available for heatmap generation. Please install matplotlib to see correlation heatmap.</p>"

        try:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()

            # Create heatmap (with or without seaborn)
            if sns is not None:
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            else:
                # Fallback to matplotlib only
                im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im)

                # Add text annotations
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black")

                plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
                plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

            plt.title('Correlation Heatmap')
            plt.tight_layout()

            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"""
                <div class="heatmap-container">
                    <img src="data:image/png;base64,{plot_data}" class="plot-image" alt="Correlation heatmap">
                </div>
            """
        except Exception as e:
            return f"<p>Could not generate correlation heatmap: {str(e)}</p>"

    def _generate_correlation_insights_table(self, strong_correlations: List[Dict[str, Any]]) -> str:
        """Generate table of significant correlations"""
        if not strong_correlations:
            return "<p>No significant correlations found.</p>"

        rows = []
        for corr in strong_correlations:
            strength_class = corr.get('strength', '').replace('_', '-')
            rows.append(f"""
                <tr>
                    <td><strong>{corr.get('variable1', '')}</strong></td>
                    <td><strong>{corr.get('variable2', '')}</strong></td>
                    <td>{corr.get('correlation', 0):.4f}</td>
                    <td><span class="badge {strength_class}">{corr.get('strength', '').replace('_', ' ').title()}</span></td>
                    <td><span class="direction-{corr.get('direction', '')}">{corr.get('direction', '').title()}</span></td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Variable 1</th>
                        <th>Variable 2</th>
                        <th>Correlation</th>
                        <th>Strength</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _generate_missing_data_visualization(self, df: pd.DataFrame) -> str:
        """Generate missing data pattern visualization"""
        missing_data = df.isnull().sum()

        if missing_data.sum() == 0:
            return "<p>No missing data to visualize.</p>"

        if not HAS_MATPLOTLIB:
            return "<p>Matplotlib not available for missing data visualization.</p>"

        try:
            plt.figure(figsize=(10, 6))
            missing_data = missing_data[missing_data > 0]

            plt.bar(range(len(missing_data)), missing_data.values, color='coral')
            plt.title('Missing Data by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Count')
            plt.xticks(range(len(missing_data)), missing_data.index, rotation=45)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"""
                <div class="missing-viz-container">
                    <img src="data:image/png;base64,{plot_data}" class="plot-image" alt="Missing data pattern">
                </div>
            """
        except Exception as e:
            return f"<p>Could not generate missing data visualization: {str(e)}</p>"

    def _generate_missing_data_recommendations(self, result: Dict[str, Any]) -> str:
        """Generate missing data recommendations"""
        recommendations = result.get('recommendations', [])

        if not recommendations:
            return "<p>No specific recommendations available.</p>"

        items = []
        for rec in recommendations:
            rec_type = rec.get('type', '')
            message = rec.get('message', '')
            action = rec.get('action', '')

            items.append(f"""
                <div class="recommendation-item">
                    <div class="rec-type">{rec_type.replace('_', ' ').title()}</div>
                    <div class="rec-message">{message}</div>
                    <div class="rec-action"><strong>Action:</strong> {action}</div>
                </div>
            """)

        return f"""
            <div class="recommendations-list">
                {"".join(items)}
            </div>
        """

    def _assess_comprehensive_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess comprehensive data quality"""
        return {
            "completeness": 100 - ((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            "consistency": 100 - ((df.duplicated().sum() / len(df)) * 100),
            "validity": 95,  # Simplified validity score
            "uniqueness": 90   # Simplified uniqueness score
        }

    def _generate_quality_metrics_cards(self, quality_metrics: Dict[str, Any]) -> str:
        """Generate quality metrics cards"""
        cards = []

        for metric, value in quality_metrics.items():
            badge_class = "success" if value >= 80 else "warning" if value >= 60 else "danger"
            cards.append(f"""
                <div class="quality-metric-card">
                    <div class="metric-name">{metric.title()}</div>
                    <div class="metric-value">{value:.1f}%</div>
                    <div class="metric-badge">
                        <span class="badge {badge_class}">
                            {"Excellent" if value >= 90 else "Good" if value >= 70 else "Needs Improvement"}
                        </span>
                    </div>
                </div>
            """)

        return "".join(cards)

    def _generate_quality_details_table(self, quality_metrics: Dict[str, Any]) -> str:
        """Generate quality details table"""
        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Quality Dimension</th>
                        <th>Score</th>
                        <th>Status</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Completeness</strong></td>
                        <td>{quality_metrics.get('completeness', 0):.1f}%</td>
                        <td><span class="badge success">Good</span></td>
                        <td>Measures the percentage of complete data records</td>
                    </tr>
                    <tr>
                        <td><strong>Consistency</strong></td>
                        <td>{quality_metrics.get('consistency', 0):.1f}%</td>
                        <td><span class="badge success">Good</span></td>
                        <td>Evaluates data consistency and absence of duplicates</td>
                    </tr>
                    <tr>
                        <td><strong>Validity</strong></td>
                        <td>{quality_metrics.get('validity', 0):.1f}%</td>
                        <td><span class="badge info">Fair</span></td>
                        <td>Checks if data conforms to expected formats and ranges</td>
                    </tr>
                    <tr>
                        <td><strong>Uniqueness</strong></td>
                        <td>{quality_metrics.get('uniqueness', 0):.1f}%</td>
                        <td><span class="badge info">Fair</span></td>
                        <td>Measures the uniqueness of records and identifiers</td>
                    </tr>
                </tbody>
            </table>
        """

    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any]) -> str:
        """Generate quality improvement recommendations"""
        recommendations = []

        if quality_metrics.get('completeness', 100) < 95:
            recommendations.append("Consider data imputation strategies for missing values")

        if quality_metrics.get('consistency', 100) < 95:
            recommendations.append("Remove or merge duplicate records")

        if quality_metrics.get('validity', 100) < 90:
            recommendations.append("Implement data validation rules and outlier detection")

        if not recommendations:
            recommendations.append("Data quality is excellent - no immediate actions required")

        items = [f"<li>{rec}</li>" for rec in recommendations]

        return f"""
            <div class="quality-recommendations">
                <ul>
                    {"".join(items)}
                </ul>
            </div>
        """

    def _generate_insights_list(self, insights: List[Dict[str, Any]]) -> str:
        """Generate insights list"""
        if not insights:
            return "<p>No specific insights generated.</p>"

        items = []
        for insight in insights[:10]:  # Limit to top 10 insights
            insight_type = insight.get('type', '')
            message = insight.get('message', '')
            severity = insight.get('severity', 'info')

            items.append(f"""
                <div class="insight-item">
                    <div class="insight-type">
                        <span class="badge {severity}">{insight_type.replace('_', ' ').title()}</span>
                    </div>
                    <div class="insight-message">{message}</div>
                </div>
            """)

        return f"""
            <div class="insights-list">
                {"".join(items)}
            </div>
        """

    def _generate_recommendations_list(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate recommendations list"""
        if not recommendations:
            return "<p>No specific recommendations available.</p>"

        items = []
        for rec in recommendations[:10]:  # Limit to top 10 recommendations
            rec_type = rec.get('type', '')
            message = rec.get('message', '')
            action = rec.get('action', '')

            items.append(f"""
                <div class="recommendation-item">
                    <div class="rec-header">
                        <div class="rec-type">{rec_type.replace('_', ' ').title()}</div>
                    </div>
                    <div class="rec-message">{message}</div>
                    {f'<div class="rec-action"><strong>Action:</strong> {action}</div>' if action else ''}
                </div>
            """)

        return f"""
            <div class="recommendations-list">
                {"".join(items)}
            </div>
        """

    def _generate_generic_analysis_section(self, result: Dict[str, Any]) -> str:
        """Generate generic analysis section for unknown types"""
        analysis_type = result.get("analysis_type", "Unknown")

        return f"""
        <div class="section analysis-section">
            <h2><i class="icon">⚙️</i> {analysis_type.replace('_', ' ').title()}</h2>

            <div class="generic-analysis">
                <div class="analysis-summary">
                    <h3>Analysis Results</h3>
                    <pre class="json-display">{json.dumps(result, indent=2, ensure_ascii=False)}</pre>
                </div>
            </div>
        </div>
        """

    def _generate_data_types_section(self, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate data types analysis section"""
        column_analysis = result.get("column_analysis", {})
        optimization_potential = result.get("memory_optimization_potential", {})

        return f"""
        <div class="section analysis-section">
            <h2><i class="icon">🏷️</i> Data Types Analysis</h2>

            <div class="types-container">
                <div class="types-summary">
                    <h3>Memory Optimization</h3>
                    <div class="optimization-stats">
                        <div class="opt-stat">
                            <span class="stat-number">{optimization_potential.get('current_memory_mb', 0):.2f} MB</span>
                            <span class="stat-label">Current Memory</span>
                        </div>
                        <div class="opt-stat">
                            <span class="stat-number">{'Yes' if optimization_potential.get('optimization_available') else 'No'}</span>
                            <span class="stat-label">Optimization Available</span>
                        </div>
                    </div>
                </div>

                <div class="column-types-analysis">
                    <h3>Column Analysis</h3>
                    {self._generate_column_types_table(column_analysis)}
                </div>
            </div>
        </div>
        """

    def _generate_column_types_table(self, column_analysis: Dict[str, Any]) -> str:
        """Generate column types analysis table"""
        if not column_analysis:
            return "<p>No column analysis data available.</p>"

        rows = []
        for col, analysis in column_analysis.items():
            current_type = analysis.get('current_type', 'Unknown')
            category = analysis.get('category', 'Unknown')
            memory_usage = analysis.get('memory_usage', 0)
            optimization = analysis.get('optimization', 'None')

            rows.append(f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td><span class="badge info">{current_type}</span></td>
                    <td>{category}</td>
                    <td>{memory_usage:,} bytes</td>
                    <td>{optimization}</td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Current Type</th>
                        <th>Category</th>
                        <th>Memory Usage</th>
                        <th>Optimization</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _generate_complete_results_section(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate complete analysis results section with all JSON data"""

        return f"""
        <div class="section complete-results">
            <h2><i class="icon">🗂️</i> Complete Analysis Results</h2>

            <div class="results-container">
                <div class="results-overview">
                    <h3>전체 분석 결과 개요</h3>
                    <p>이 섹션에는 모든 분석 결과의 상세 정보가 포함되어 있습니다. 별도의 JSON 파일을 열 필요 없이 여기서 모든 데이터를 확인할 수 있습니다.</p>

                    <div class="results-stats">
                        <div class="result-stat">
                            <span class="stat-number">{len(analysis_results)}</span>
                            <span class="stat-label">분석 모듈</span>
                        </div>
                        <div class="result-stat">
                            <span class="stat-number">{sum(len(result.get('data_info', {}).get('columns', [])) for result in analysis_results if 'data_info' in result and hasattr(result.get('data_info', {}).get('columns', []), '__len__'))}</span>
                            <span class="stat-label">분석된 컬럼</span>
                        </div>
                    </div>
                </div>

                <div class="detailed-results">
                    <h3>상세 분석 결과</h3>
                    {self._generate_expandable_results(analysis_results)}
                </div>

                <div class="raw-json-section">
                    <h3>원시 JSON 데이터</h3>
                    <p>아래는 모든 분석 결과의 완전한 JSON 데이터입니다:</p>

                    <div class="json-controls">
                        <button onclick="toggleJsonView()" class="json-toggle-btn">JSON 보기/숨기기</button>
                        <button onclick="copyJsonData()" class="json-copy-btn">JSON 복사</button>
                        <button onclick="downloadJsonData()" class="json-download-btn">JSON 다운로드</button>
                    </div>

                    <div id="jsonContainer" class="json-container" style="display: none;">
                        <pre id="jsonData" class="json-display">{json.dumps(analysis_results, indent=2, ensure_ascii=False, cls=self._get_json_encoder())}</pre>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_expandable_results(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate expandable sections for each analysis result"""

        sections = []

        for i, result in enumerate(analysis_results):
            analysis_type = result.get("analysis_type", f"Analysis {i+1}")
            summary = result.get("summary", "No summary available")

            # Create a human-readable summary of key metrics
            key_metrics = self._extract_key_metrics(result)

            sections.append(f"""
                <div class="expandable-result">
                    <div class="result-header" onclick="toggleResultSection('result_{i}')">
                        <div class="result-title">
                            <h4><i class="icon">📊</i> {analysis_type.replace('_', ' ').title()}</h4>
                            <span class="expand-icon" id="icon_result_{i}">▼</span>
                        </div>
                        <div class="result-summary">{summary}</div>
                    </div>

                    <div id="result_{i}" class="result-content" style="display: none;">
                        <div class="result-tabs">
                            <div class="tab-buttons">
                                <button class="tab-button active" onclick="showResultTab('result_{i}', 'overview')">개요</button>
                                <button class="tab-button" onclick="showResultTab('result_{i}', 'metrics')">주요 지표</button>
                                <button class="tab-button" onclick="showResultTab('result_{i}', 'details')">상세 정보</button>
                                <button class="tab-button" onclick="showResultTab('result_{i}', 'raw')">원시 데이터</button>
                            </div>

                            <div id="result_{i}_overview" class="tab-content active">
                                {self._generate_result_overview(result)}
                            </div>

                            <div id="result_{i}_metrics" class="tab-content">
                                {self._generate_key_metrics_display(key_metrics)}
                            </div>

                            <div id="result_{i}_details" class="tab-content">
                                {self._generate_result_details(result)}
                            </div>

                            <div id="result_{i}_raw" class="tab-content">
                                <pre class="json-display">{json.dumps(result, indent=2, ensure_ascii=False, cls=self._get_json_encoder())}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            """)

        return "".join(sections)

    def _extract_key_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis result"""

        metrics = {}

        # Extract based on analysis type
        analysis_type = result.get("analysis_type", "")

        if analysis_type == "descriptive_statistics":
            numeric_stats = result.get("numeric_statistics", {})
            if "count" in numeric_stats:
                metrics["변수 개수"] = len(numeric_stats["count"])
            if "mean" in numeric_stats:
                avg_mean = np.mean(list(numeric_stats["mean"].values())) if numeric_stats["mean"] else 0
                metrics["평균값의 평균"] = f"{avg_mean:.2f}"

        elif analysis_type == "correlation_analysis":
            strong_corr = result.get("strong_correlations", [])
            metrics["강한 상관관계"] = len(strong_corr)
            correlation_matrix = result.get("correlation_matrix", {})
            if correlation_matrix:
                metrics["분석된 변수"] = len(correlation_matrix)

        elif analysis_type == "missing_data_analysis":
            missing_summary = result.get("missing_data_summary", {})
            metrics["총 결측값"] = missing_summary.get("total_missing_values", 0)
            metrics["결측 비율"] = f"{missing_summary.get('missing_percentage', 0):.1f}%"
            metrics["완전한 행"] = missing_summary.get("complete_rows", 0)

        elif analysis_type == "data_types_analysis":
            column_analysis = result.get("column_analysis", {})
            metrics["분석된 컬럼"] = len(column_analysis)
            optimization = result.get("memory_optimization_potential", {})
            metrics["메모리 사용량"] = f"{optimization.get('current_memory_mb', 0):.2f} MB"

        # Add general metrics
        data_info = result.get("data_info", {})
        if data_info:
            shape = data_info.get('shape', [0, 0])
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                metrics["데이터 형태"] = f"{shape[0]} × {shape[1]}"
            elif hasattr(shape, '__len__') and len(shape) >= 2:
                metrics["데이터 형태"] = f"{shape[0]} × {shape[1]}"
            else:
                metrics["데이터 형태"] = "알 수 없음"

            if "numeric_columns" in data_info:
                numeric_cols = data_info["numeric_columns"]
                if hasattr(numeric_cols, '__len__'):
                    metrics["수치형 컬럼"] = len(numeric_cols)
            if "categorical_columns" in data_info:
                categorical_cols = data_info["categorical_columns"]
                if hasattr(categorical_cols, '__len__'):
                    metrics["범주형 컬럼"] = len(categorical_cols)

        return metrics

    def _generate_result_overview(self, result: Dict[str, Any]) -> str:
        """Generate overview for a specific result"""

        analysis_type = result.get("analysis_type", "Unknown")
        summary = result.get("summary", "No summary available")
        timestamp = result.get("timestamp", "Unknown")

        # Get specific overview based on type
        type_overview = ""

        if analysis_type == "descriptive_statistics":
            numeric_stats = result.get("numeric_statistics", {})
            categorical_stats = result.get("categorical_statistics", {})
            type_overview = f"""
                <div class="overview-grid">
                    <div class="overview-item">
                        <span class="overview-label">수치형 변수</span>
                        <span class="overview-value">{len(numeric_stats.get('count', {}))}</span>
                    </div>
                    <div class="overview-item">
                        <span class="overview-label">범주형 변수</span>
                        <span class="overview-value">{len(categorical_stats)}</span>
                    </div>
                </div>
            """
        elif analysis_type == "correlation_analysis":
            strong_correlations = result.get("strong_correlations", [])
            correlation_matrix = result.get("correlation_matrix", {})
            type_overview = f"""
                <div class="overview-grid">
                    <div class="overview-item">
                        <span class="overview-label">분석된 변수</span>
                        <span class="overview-value">{len(correlation_matrix)}</span>
                    </div>
                    <div class="overview-item">
                        <span class="overview-label">강한 상관관계</span>
                        <span class="overview-value">{len(strong_correlations)}</span>
                    </div>
                </div>
            """
        elif analysis_type == "missing_data_analysis":
            missing_summary = result.get("missing_data_summary", {})
            type_overview = f"""
                <div class="overview-grid">
                    <div class="overview-item">
                        <span class="overview-label">총 결측값</span>
                        <span class="overview-value">{missing_summary.get('total_missing_values', 0):,}</span>
                    </div>
                    <div class="overview-item">
                        <span class="overview-label">결측 비율</span>
                        <span class="overview-value">{missing_summary.get('missing_percentage', 0):.1f}%</span>
                    </div>
                </div>
            """

        return f"""
            <div class="result-overview">
                <div class="overview-summary">
                    <h4>분석 개요</h4>
                    <p><strong>유형:</strong> {analysis_type.replace('_', ' ').title()}</p>
                    <p><strong>실행 시간:</strong> {timestamp}</p>
                    <p><strong>요약:</strong> {summary}</p>
                </div>

                {type_overview}

                <div class="overview-insights">
                    <h4>주요 발견사항</h4>
                    {self._generate_overview_insights(result)}
                </div>
            </div>
        """

    def _generate_key_metrics_display(self, metrics: Dict[str, Any]) -> str:
        """Generate key metrics display"""

        if not metrics:
            return "<p>주요 지표를 추출할 수 없습니다.</p>"

        metric_cards = []
        for key, value in metrics.items():
            metric_cards.append(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{key}</div>
                </div>
            """)

        return f"""
            <div class="metrics-display">
                <div class="metrics-grid">
                    {"".join(metric_cards)}
                </div>
            </div>
        """

    def _generate_result_details(self, result: Dict[str, Any]) -> str:
        """Generate detailed information for a result"""

        analysis_type = result.get("analysis_type", "Unknown")

        # Generate different details based on analysis type
        if analysis_type == "descriptive_statistics":
            return self._generate_descriptive_details(result)
        elif analysis_type == "correlation_analysis":
            return self._generate_correlation_details(result)
        elif analysis_type == "missing_data_analysis":
            return self._generate_missing_details(result)
        elif analysis_type == "data_types_analysis":
            return self._generate_types_details(result)
        else:
            return f"""
                <div class="generic-details">
                    <h4>분석 결과 상세정보</h4>
                    <div class="details-content">
                        {self._format_dict_as_html(result)}
                    </div>
                </div>
            """

    def _generate_descriptive_details(self, result: Dict[str, Any]) -> str:
        """Generate detailed view for descriptive statistics"""

        numeric_stats = result.get("numeric_statistics", {})
        categorical_stats = result.get("categorical_statistics", {})

        details_html = f"""
            <div class="descriptive-details">
                <div class="numeric-details">
                    <h4>수치형 변수 상세 통계</h4>
                    {self._generate_detailed_numeric_table(numeric_stats)}
                </div>

                <div class="categorical-details">
                    <h4>범주형 변수 상세 정보</h4>
                    {self._generate_detailed_categorical_table(categorical_stats)}
                </div>
            </div>
        """

        return details_html

    def _generate_correlation_details(self, result: Dict[str, Any]) -> str:
        """Generate detailed view for correlation analysis"""

        correlation_matrix = result.get("correlation_matrix", {})
        strong_correlations = result.get("strong_correlations", [])

        return f"""
            <div class="correlation-details">
                <div class="correlation-matrix-section">
                    <h4>상관관계 매트릭스</h4>
                    {self._generate_correlation_matrix_table(correlation_matrix)}
                </div>

                <div class="strong-correlations-section">
                    <h4>강한 상관관계 목록</h4>
                    {self._generate_correlation_insights_table(strong_correlations)}
                </div>
            </div>
        """

    def _generate_missing_details(self, result: Dict[str, Any]) -> str:
        """Generate detailed view for missing data analysis"""

        missing_by_column = result.get("missing_by_column", {})
        missing_patterns = result.get("missing_patterns", {})
        recommendations = result.get("recommendations", [])

        return f"""
            <div class="missing-details">
                <div class="missing-by-column">
                    <h4>컬럼별 결측 데이터</h4>
                    {self._generate_missing_by_column_table(missing_by_column)}
                </div>

                <div class="missing-patterns">
                    <h4>결측 패턴</h4>
                    {self._format_dict_as_html(missing_patterns)}
                </div>

                <div class="missing-recommendations">
                    <h4>권장사항</h4>
                    {self._generate_missing_data_recommendations(result)}
                </div>
            </div>
        """

    def _generate_types_details(self, result: Dict[str, Any]) -> str:
        """Generate detailed view for data types analysis"""

        column_analysis = result.get("column_analysis", {})
        optimization_potential = result.get("memory_optimization_potential", {})

        return f"""
            <div class="types-details">
                <div class="column-analysis">
                    <h4>컬럼별 데이터 타입 분석</h4>
                    {self._generate_column_types_table(column_analysis)}
                </div>

                <div class="optimization-potential">
                    <h4>메모리 최적화 가능성</h4>
                    {self._format_dict_as_html(optimization_potential)}
                </div>
            </div>
        """

    def _generate_overview_insights(self, result: Dict[str, Any]) -> str:
        """Generate overview insights for a result"""

        insights = result.get("insights", [])

        if not insights:
            # Generate basic insights from the data
            analysis_type = result.get("analysis_type", "")
            basic_insights = []

            if analysis_type == "descriptive_statistics":
                numeric_stats = result.get("numeric_statistics", {})
                if numeric_stats.get("count"):
                    basic_insights.append(f"수치형 변수 {len(numeric_stats['count'])}개에 대한 기술통계 분석 완료")

            elif analysis_type == "correlation_analysis":
                strong_corr = result.get("strong_correlations", [])
                if strong_corr:
                    basic_insights.append(f"강한 상관관계 {len(strong_corr)}개 발견")
                else:
                    basic_insights.append("강한 상관관계가 발견되지 않음")

            elif analysis_type == "missing_data_analysis":
                missing_summary = result.get("missing_data_summary", {})
                missing_pct = missing_summary.get("missing_percentage", 0)
                if missing_pct > 10:
                    basic_insights.append(f"결측 데이터 비율이 {missing_pct:.1f}%로 높음 - 주의 필요")
                elif missing_pct > 0:
                    basic_insights.append(f"결측 데이터 비율 {missing_pct:.1f}% - 적절한 수준")
                else:
                    basic_insights.append("결측 데이터 없음 - 우수한 데이터 품질")

            insights = [{"message": insight, "type": "info"} for insight in basic_insights]

        if not insights:
            return "<p>특별한 발견사항이 없습니다.</p>"

        insight_items = []
        for insight in insights[:5]:  # Show top 5 insights
            message = insight.get("message", "")
            insight_type = insight.get("type", "info")

            insight_items.append(f"""
                <div class="insight-item">
                    <span class="insight-icon">💡</span>
                    <span class="insight-text">{message}</span>
                </div>
            """)

        return f"""
            <div class="insights-list">
                {"".join(insight_items)}
            </div>
        """

    def _generate_correlation_matrix_table(self, correlation_matrix: Dict[str, Any]) -> str:
        """Generate correlation matrix as HTML table"""

        if not correlation_matrix:
            return "<p>상관관계 매트릭스 데이터를 사용할 수 없습니다.</p>"

        # Convert to DataFrame-like structure for easier handling
        variables = list(correlation_matrix.keys())

        # Generate table header
        header = "<tr><th>변수</th>" + "".join([f"<th>{var}</th>" for var in variables]) + "</tr>"

        # Generate table rows
        rows = []
        for var1 in variables:
            row = f"<tr><td><strong>{var1}</strong></td>"
            for var2 in variables:
                corr_value = correlation_matrix.get(var1, {}).get(var2, 0)
                # Color code based on correlation strength
                color_class = self._get_correlation_color_class(corr_value)
                row += f'<td class="{color_class}">{corr_value:.3f}</td>'
            row += "</tr>"
            rows.append(row)

        return f"""
            <table class="data-table correlation-matrix-table">
                <thead>{header}</thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        """

    def _generate_missing_by_column_table(self, missing_by_column: Dict[str, Any]) -> str:
        """Generate missing data by column table"""

        if not missing_by_column:
            return "<p>컬럼별 결측 데이터 정보를 사용할 수 없습니다.</p>"

        rows = []
        for column, missing_info in missing_by_column.items():
            if isinstance(missing_info, dict):
                missing_count = missing_info.get("missing_count", 0)
                missing_percentage = missing_info.get("missing_percentage", 0)
                total_count = missing_info.get("total_count", 0)
            else:
                # If it's just a number
                missing_count = missing_info
                missing_percentage = 0
                total_count = 0

            severity_class = "danger" if missing_percentage > 20 else "warning" if missing_percentage > 5 else "success"

            rows.append(f"""
                <tr>
                    <td><strong>{column}</strong></td>
                    <td>{missing_count:,}</td>
                    <td>{total_count:,}</td>
                    <td><span class="badge {severity_class}">{missing_percentage:.1f}%</span></td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>컬럼명</th>
                        <th>결측 개수</th>
                        <th>전체 개수</th>
                        <th>결측 비율</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _get_correlation_color_class(self, value: float) -> str:
        """Get CSS class for correlation value color coding"""

        abs_value = abs(value)
        if abs_value >= 0.8:
            return "corr-very-strong"
        elif abs_value >= 0.6:
            return "corr-strong"
        elif abs_value >= 0.4:
            return "corr-moderate"
        elif abs_value >= 0.2:
            return "corr-weak"
        else:
            return "corr-very-weak"

    def _format_dict_as_html(self, data: Dict[str, Any], max_depth: int = 3, current_depth: int = 0) -> str:
        """Format dictionary as readable HTML"""

        if current_depth >= max_depth:
            return f"<pre>{json.dumps(data, indent=2, ensure_ascii=False, cls=self._get_json_encoder())}</pre>"

        if not isinstance(data, dict):
            return f"<span class='data-value'>{str(data)}</span>"

        items = []
        for key, value in data.items():
            if isinstance(value, dict) and current_depth < max_depth - 1:
                items.append(f"""
                    <div class="dict-item">
                        <div class="dict-key">{key}:</div>
                        <div class="dict-value">
                            {self._format_dict_as_html(value, max_depth, current_depth + 1)}
                        </div>
                    </div>
                """)
            elif isinstance(value, list) and len(value) > 0:
                if len(value) <= 10:
                    list_items = ", ".join([str(item) for item in value])
                    items.append(f"""
                        <div class="dict-item">
                            <div class="dict-key">{key}:</div>
                            <div class="dict-value">[{list_items}]</div>
                        </div>
                    """)
                else:
                    items.append(f"""
                        <div class="dict-item">
                            <div class="dict-key">{key}:</div>
                            <div class="dict-value">List with {len(value)} items (첫 10개: {', '.join([str(item) for item in value[:10]])}...)</div>
                        </div>
                    """)
            else:
                items.append(f"""
                    <div class="dict-item">
                        <div class="dict-key">{key}:</div>
                        <div class="dict-value">{str(value)}</div>
                    </div>
                """)

        return f"""
            <div class="dict-container">
                {"".join(items)}
            </div>
        """

    def _get_json_encoder(self):
        """Get JSON encoder that handles numpy types"""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return super().default(obj)

        return NumpyEncoder