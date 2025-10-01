#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Assessment for Lightweight Analysis MCP
경량 분석 MCP용 데이터 품질 평가 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # Fallback implementations
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def assess_data_quality_comprehensive(df: pd.DataFrame, generate_report: bool = False) -> Dict[str, Any]:
    """
    Comprehensive data quality assessment
    포괄적인 데이터 품질 평가
    """
    results = {}

    # 1. Completeness Assessment (완전성 평가)
    completeness_scores = assess_completeness(df)
    results["completeness"] = completeness_scores

    # 2. Consistency Assessment (일관성 평가)
    consistency_scores = assess_consistency(df)
    results["consistency"] = consistency_scores

    # 3. Accuracy Assessment (정확성 평가)
    accuracy_scores = assess_accuracy(df)
    results["accuracy"] = accuracy_scores

    # 4. Validity Assessment (유효성 평가)
    validity_scores = assess_validity(df)
    results["validity"] = validity_scores

    # 5. Overall Quality Scores (전체 품질 점수)
    quality_scores = calculate_overall_quality_scores(
        completeness_scores, consistency_scores, accuracy_scores, validity_scores
    )
    results["quality_scores"] = quality_scores

    # 6. Data Quality Issues (데이터 품질 이슈)
    quality_issues = identify_quality_issues(df, results)
    results["quality_issues"] = quality_issues

    # 7. Improvement Recommendations (개선 권장사항)
    recommendations = generate_improvement_recommendations(results, df)
    results["recommendations"] = recommendations

    # 8. Overall Score and Grade
    overall_score = quality_scores["overall_score"]
    results["overall_score"] = overall_score
    results["quality_grade"] = classify_quality_grade(overall_score)

    # 9. Detailed Report (if requested)
    if generate_report:
        detailed_report = generate_detailed_quality_report(df, results)
        results["detailed_report"] = detailed_report

    return results


def assess_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess data completeness
    데이터 완전성 평가
    """
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness_percentage = ((total_cells - missing_cells) / total_cells) * 100

    # Column-wise completeness
    column_completeness = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        col_completeness = ((len(df) - missing_count) / len(df)) * 100
        column_completeness[col] = {
            "completeness_percentage": round(col_completeness, 2),
            "missing_count": int(missing_count),
            "status": classify_completeness_status(col_completeness)
        }

    # Row-wise completeness
    complete_rows = len(df) - df.isnull().any(axis=1).sum()
    row_completeness_percentage = (complete_rows / len(df)) * 100

    return {
        "overall_completeness": round(completeness_percentage, 2),
        "row_completeness": round(row_completeness_percentage, 2),
        "column_completeness": column_completeness,
        "complete_rows": int(complete_rows),
        "incomplete_rows": int(len(df) - complete_rows),
        "score": min(100, completeness_percentage)  # Score capped at 100
    }


def classify_completeness_status(completeness_pct: float) -> str:
    """Classify completeness status"""
    if completeness_pct >= 95:
        return "우수"
    elif completeness_pct >= 85:
        return "양호"
    elif completeness_pct >= 70:
        return "보통"
    else:
        return "개선필요"


def assess_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess data consistency
    데이터 일관성 평가
    """
    consistency_issues = []
    consistency_score = 100

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        consistency_issues.append({
            "type": "중복 행",
            "count": int(duplicate_rows),
            "percentage": round(duplicate_percentage, 2),
            "severity": "높음" if duplicate_percentage > 5 else "보통"
        })
        consistency_score -= min(20, duplicate_percentage * 2)

    # Check for data type inconsistencies
    type_inconsistencies = check_data_type_consistency(df)
    if type_inconsistencies:
        consistency_issues.extend(type_inconsistencies)
        consistency_score -= len(type_inconsistencies) * 5

    # Check for format inconsistencies in string columns
    format_inconsistencies = check_format_consistency(df)
    if format_inconsistencies:
        consistency_issues.extend(format_inconsistencies)
        consistency_score -= len(format_inconsistencies) * 3

    # Check for value range inconsistencies
    range_inconsistencies = check_value_range_consistency(df)
    if range_inconsistencies:
        consistency_issues.extend(range_inconsistencies)
        consistency_score -= len(range_inconsistencies) * 5

    return {
        "score": max(0, consistency_score),
        "issues": consistency_issues,
        "duplicate_rows": int(duplicate_rows),
        "status": classify_score_status(max(0, consistency_score))
    }


def check_data_type_consistency(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for data type inconsistencies"""
    inconsistencies = []

    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric-looking strings exist
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Try to convert to numeric
                numeric_convertible = pd.to_numeric(non_null_values, errors='coerce').notna().sum()
                total_values = len(non_null_values)

                if 0 < numeric_convertible < total_values:
                    mixed_percentage = ((total_values - numeric_convertible) / total_values) * 100
                    if mixed_percentage < 90:  # If more than 10% are numeric
                        inconsistencies.append({
                            "type": "데이터 타입 혼재",
                            "column": col,
                            "description": f"문자열과 숫자 값이 혼재됨",
                            "numeric_percentage": round((numeric_convertible / total_values) * 100, 2),
                            "severity": "보통"
                        })

    return inconsistencies


def check_format_consistency(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for format inconsistencies in string columns"""
    inconsistencies = []

    string_cols = df.select_dtypes(include=['object']).columns

    for col in string_cols:
        non_null_values = df[col].dropna().astype(str)
        if len(non_null_values) == 0:
            continue

        # Check for inconsistent spacing
        has_leading_spaces = non_null_values.str.startswith(' ').any()
        has_trailing_spaces = non_null_values.str.endswith(' ').any()

        if has_leading_spaces or has_trailing_spaces:
            inconsistencies.append({
                "type": "공백 문제",
                "column": col,
                "description": "앞뒤 공백이 일관되지 않음",
                "severity": "낮음"
            })

        # Check for case inconsistencies
        if len(non_null_values.unique()) != len(non_null_values.str.lower().unique()):
            inconsistencies.append({
                "type": "대소문자 불일치",
                "column": col,
                "description": "대소문자 사용이 일관되지 않음",
                "severity": "낮음"
            })

    return inconsistencies


def check_value_range_consistency(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for value range inconsistencies"""
    inconsistencies = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        clean_values = df[col].dropna()
        if len(clean_values) == 0:
            continue

        # Check for extreme outliers (beyond 3 standard deviations)
        mean_val = clean_values.mean()
        std_val = clean_values.std()

        if std_val > 0:
            outliers = clean_values[(clean_values < mean_val - 3*std_val) |
                                   (clean_values > mean_val + 3*std_val)]
            outlier_percentage = (len(outliers) / len(clean_values)) * 100

            if outlier_percentage > 5:  # More than 5% outliers
                inconsistencies.append({
                    "type": "극값 이상치",
                    "column": col,
                    "description": f"통계적 이상치가 {outlier_percentage:.1f}% 존재",
                    "outlier_count": len(outliers),
                    "severity": "보통" if outlier_percentage > 10 else "낮음"
                })

    return inconsistencies


def assess_accuracy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess data accuracy
    데이터 정확성 평가
    """
    accuracy_issues = []
    accuracy_score = 100

    # Check for logical inconsistencies
    logical_issues = check_logical_consistency(df)
    if logical_issues:
        accuracy_issues.extend(logical_issues)
        accuracy_score -= len(logical_issues) * 8

    # Check for impossible values
    impossible_values = check_impossible_values(df)
    if impossible_values:
        accuracy_issues.extend(impossible_values)
        accuracy_score -= len(impossible_values) * 10

    # Check for suspicious patterns
    suspicious_patterns = detect_suspicious_patterns(df)
    if suspicious_patterns:
        accuracy_issues.extend(suspicious_patterns)
        accuracy_score -= len(suspicious_patterns) * 5

    return {
        "score": max(0, accuracy_score),
        "issues": accuracy_issues,
        "status": classify_score_status(max(0, accuracy_score))
    }


def check_logical_consistency(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for logical inconsistencies"""
    issues = []

    # Check for negative values where they shouldn't be (e.g., age, count)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['age', 'count', 'quantity', 'amount', 'price', 'weight', 'height']):
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append({
                    "type": "논리적 불일치",
                    "column": col,
                    "description": f"음수 값이 존재하는 컬럼 ({negative_count}개)",
                    "severity": "높음"
                })

    return issues


def check_impossible_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for impossible values"""
    issues = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Check for age-related impossible values
        if 'age' in col.lower():
            impossible_age = ((df[col] < 0) | (df[col] > 150)).sum()
            if impossible_age > 0:
                issues.append({
                    "type": "불가능한 값",
                    "column": col,
                    "description": f"불가능한 나이 값 ({impossible_age}개: 0세 미만 또는 150세 초과)",
                    "severity": "높음"
                })

        # Check for percentage values outside 0-100
        if any(keyword in col.lower() for keyword in ['percentage', 'percent', 'rate', 'ratio']):
            impossible_pct = ((df[col] < 0) | (df[col] > 100)).sum()
            if impossible_pct > 0:
                issues.append({
                    "type": "불가능한 값",
                    "column": col,
                    "description": f"잘못된 비율 값 ({impossible_pct}개: 0-100 범위 외)",
                    "severity": "높음"
                })

    return issues


def detect_suspicious_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect suspicious patterns in data"""
    issues = []

    # Check for too many repeated values
    for col in df.columns:
        if df[col].notna().sum() > 0:
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                most_frequent_count = value_counts.iloc[0]
                total_non_null = df[col].notna().sum()
                frequency_ratio = most_frequent_count / total_non_null

                if frequency_ratio > 0.9 and len(value_counts) > 1:  # 90% same value
                    issues.append({
                        "type": "의심스러운 패턴",
                        "column": col,
                        "description": f"값의 {frequency_ratio*100:.1f}%가 동일함 ('{value_counts.index[0]}')",
                        "severity": "보통"
                    })

    return issues


def assess_validity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess data validity
    데이터 유효성 평가
    """
    validity_issues = []
    validity_score = 100

    # Check for invalid formats
    format_issues = check_data_formats(df)
    if format_issues:
        validity_issues.extend(format_issues)
        validity_score -= len(format_issues) * 7

    # Check for constraint violations
    constraint_issues = check_business_rules(df)
    if constraint_issues:
        validity_issues.extend(constraint_issues)
        validity_score -= len(constraint_issues) * 10

    return {
        "score": max(0, validity_score),
        "issues": validity_issues,
        "status": classify_score_status(max(0, validity_score))
    }


def check_data_formats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for invalid data formats"""
    issues = []

    for col in df.columns:
        if df[col].dtype == 'object':
            non_null_values = df[col].dropna().astype(str)
            if len(non_null_values) == 0:
                continue

            # Check for email-like columns
            if 'email' in col.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = non_null_values[~non_null_values.str.match(email_pattern, na=False)]
                if len(invalid_emails) > 0:
                    issues.append({
                        "type": "잘못된 형식",
                        "column": col,
                        "description": f"유효하지 않은 이메일 형식 ({len(invalid_emails)}개)",
                        "severity": "보통"
                    })

            # Check for phone-like columns
            if any(keyword in col.lower() for keyword in ['phone', 'tel', '전화']):
                # Simple check for phone numbers (contains only digits, spaces, hyphens, parentheses)
                phone_pattern = r'^[\d\s\-\(\)\+]+$'
                invalid_phones = non_null_values[~non_null_values.str.match(phone_pattern, na=False)]
                if len(invalid_phones) > 0:
                    issues.append({
                        "type": "잘못된 형식",
                        "column": col,
                        "description": f"유효하지 않은 전화번호 형식 ({len(invalid_phones)}개)",
                        "severity": "보통"
                    })

    return issues


def check_business_rules(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check for business rule violations"""
    issues = []

    # Example business rules - these would be customized based on domain
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Check for zero values where they might not make sense
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount', 'salary']):
            zero_count = (df[col] == 0).sum()
            if zero_count > len(df) * 0.1:  # More than 10% are zero
                issues.append({
                    "type": "비즈니스 규칙 위반",
                    "column": col,
                    "description": f"금액 관련 컬럼에 0값이 {zero_count}개 존재 (전체의 {zero_count/len(df)*100:.1f}%)",
                    "severity": "보통"
                })

    return issues


def calculate_overall_quality_scores(completeness: Dict, consistency: Dict, accuracy: Dict, validity: Dict) -> Dict[str, float]:
    """Calculate overall quality scores"""
    # Weighted average of different quality dimensions
    weights = {
        "completeness": 0.3,
        "consistency": 0.25,
        "accuracy": 0.3,
        "validity": 0.15
    }

    overall_score = (
        completeness["score"] * weights["completeness"] +
        consistency["score"] * weights["consistency"] +
        accuracy["score"] * weights["accuracy"] +
        validity["score"] * weights["validity"]
    )

    return {
        "completeness": round(completeness["score"], 1),
        "consistency": round(consistency["score"], 1),
        "accuracy": round(accuracy["score"], 1),
        "validity": round(validity["score"], 1),
        "overall_score": round(overall_score, 1)
    }


def classify_score_status(score: float) -> str:
    """Classify quality score status"""
    if score >= 90:
        return "우수"
    elif score >= 75:
        return "양호"
    elif score >= 60:
        return "보통"
    else:
        return "개선필요"


def classify_quality_grade(overall_score: float) -> str:
    """Classify overall quality grade"""
    if overall_score >= 90:
        return "A"
    elif overall_score >= 80:
        return "B"
    elif overall_score >= 70:
        return "C"
    elif overall_score >= 60:
        return "D"
    else:
        return "F"


def identify_quality_issues(df: pd.DataFrame, results: Dict[str, Any]) -> List[str]:
    """Identify top quality issues"""
    issues = []

    # Collect all issues from different assessments
    all_issues = []
    for assessment in ['completeness', 'consistency', 'accuracy', 'validity']:
        if 'issues' in results.get(assessment, {}):
            all_issues.extend(results[assessment]['issues'])

    # Sort by severity and create summary
    high_severity_issues = [issue for issue in all_issues if issue.get('severity') == '높음']
    medium_severity_issues = [issue for issue in all_issues if issue.get('severity') == '보통']

    if high_severity_issues:
        issues.append(f"심각한 데이터 품질 문제 {len(high_severity_issues)}개 발견")

    if medium_severity_issues:
        issues.append(f"중간 수준의 데이터 품질 문제 {len(medium_severity_issues)}개 발견")

    # Add specific major issues
    if results['completeness']['overall_completeness'] < 85:
        issues.append(f"전체 완전성이 낮음 ({results['completeness']['overall_completeness']:.1f}%)")

    if results['consistency'].get('duplicate_rows', 0) > 0:
        issues.append(f"중복된 행 {results['consistency']['duplicate_rows']}개 존재")

    return issues[:5]  # Return top 5 issues


def generate_improvement_recommendations(results: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate improvement recommendations"""
    recommendations = []

    # Completeness recommendations
    if results['completeness']['score'] < 85:
        incomplete_cols = [col for col, info in results['completeness']['column_completeness'].items()
                          if info['completeness_percentage'] < 80]
        if incomplete_cols:
            recommendations.append({
                "priority": "높음",
                "category": "완전성 개선",
                "action": f"높은 결측률 컬럼 처리: {', '.join(incomplete_cols[:3])}",
                "details": "대체 방법 적용 또는 추가 데이터 수집 고려"
            })

    # Consistency recommendations
    if results['consistency']['score'] < 80:
        if results['consistency'].get('duplicate_rows', 0) > 0:
            recommendations.append({
                "priority": "높음",
                "category": "일관성 개선",
                "action": f"중복 행 {results['consistency']['duplicate_rows']}개 제거",
                "details": "데이터 중복 제거 및 수집 프로세스 점검"
            })

    # Accuracy recommendations
    if results['accuracy']['score'] < 80:
        recommendations.append({
            "priority": "보통",
            "category": "정확성 개선",
            "action": "데이터 검증 규칙 강화",
            "details": "입력 단계에서 유효성 검사 및 이상치 탐지 프로세스 구축"
        })

    # Overall recommendations
    overall_score = results['quality_scores']['overall_score']
    if overall_score < 70:
        recommendations.append({
            "priority": "높음",
            "category": "전반적 개선",
            "action": "포괄적인 데이터 품질 관리 체계 구축",
            "details": "데이터 수집부터 저장까지 전체 파이프라인 점검"
        })

    return recommendations


def generate_detailed_quality_report(df: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed quality report"""
    report = {
        "executive_summary": {
            "total_records": len(df),
            "total_fields": len(df.columns),
            "overall_quality_grade": results['quality_grade'],
            "overall_score": results['overall_score'],
            "data_usability": "높음" if results['overall_score'] >= 80 else "보통" if results['overall_score'] >= 60 else "낮음"
        },
        "dimension_analysis": {
            "completeness": {
                "score": results['completeness']['score'],
                "key_finding": f"데이터 완전성 {results['completeness']['overall_completeness']:.1f}%"
            },
            "consistency": {
                "score": results['consistency']['score'],
                "key_finding": f"중복 행 {results['consistency'].get('duplicate_rows', 0)}개"
            },
            "accuracy": {
                "score": results['accuracy']['score'],
                "key_finding": f"정확성 문제 {len(results['accuracy']['issues'])}건"
            },
            "validity": {
                "score": results['validity']['score'],
                "key_finding": f"유효성 문제 {len(results['validity']['issues'])}건"
            }
        },
        "actionable_next_steps": [
            "우선순위가 높은 품질 이슈부터 해결",
            "데이터 수집 프로세스 개선 방안 수립",
            "정기적인 품질 모니터링 체계 구축"
        ]
    }

    return report


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['file_path'])

        # Load data
        df = load_data(options['file_path'])

        # Get basic data info
        data_info = get_data_info(df)

        # Perform data quality assessment
        generate_report = options.get('generate_report', False)
        analysis_results = assess_data_quality_comprehensive(df, generate_report)

        # Create final result
        result = create_analysis_result(
            analysis_type="data_quality_assessment",
            data_info=data_info,
            results=analysis_results,
            summary=f"데이터 품질 평가 완료 - 종합 점수: {analysis_results['overall_score']:.1f}점 ({analysis_results['quality_grade']}등급)"
        )

        # Output results
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "data_quality_assessment"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()