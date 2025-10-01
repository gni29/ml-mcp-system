#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Automatic Data Type Judgment & Pipeline Creation Demo
MCP 자동 데이터 타입 판단 및 파이프라인 생성 데모
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List

class AutoDataProcessor:
    """
    MCP의 자동 데이터 처리 시스템 데모
    실제 MCP에서 사용되는 자동 판단 로직
    """

    def __init__(self):
        self.column_analysis = {}
        self.preprocessing_pipeline = []
        self.analytics_pipeline = []

    def judge_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        1단계: 자동 데이터 타입 판단
        Column A: numeric, Column B: categorical 등 자동 분류
        """
        print("🔍 Step 1: Automatic Data Type Judgment")
        print("=" * 50)

        type_analysis = {}

        for column in df.columns:
            col_data = df[column]

            # 기본 타입 정보
            analysis = {
                'original_dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': round(col_data.isnull().sum() / len(col_data) * 100, 2),
                'unique_count': col_data.nunique(),
                'unique_percentage': round(col_data.nunique() / len(col_data) * 100, 2)
            }

            # 자동 타입 분류
            if pd.api.types.is_numeric_dtype(col_data):
                analysis['mcp_type'] = 'numeric'
                analysis['sub_type'] = self._classify_numeric(col_data)
                analysis.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'has_outliers': self._detect_outliers(col_data),
                    'distribution': self._analyze_distribution(col_data)
                })

            elif pd.api.types.is_object_dtype(col_data):
                analysis['mcp_type'] = 'categorical'
                analysis['sub_type'] = self._classify_categorical(col_data)

                # 빈도 분석
                value_counts = col_data.value_counts()
                analysis.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'cardinality_level': self._classify_cardinality(col_data)
                })

                # 날짜/시간 가능성 체크
                if self._might_be_datetime(col_data):
                    analysis['potential_datetime'] = True

            elif pd.api.types.is_datetime64_any_dtype(col_data):
                analysis['mcp_type'] = 'datetime'
                analysis.update({
                    'date_range': f"{col_data.min()} to {col_data.max()}",
                    'time_span_days': (col_data.max() - col_data.min()).days
                })

            elif pd.api.types.is_bool_dtype(col_data):
                analysis['mcp_type'] = 'boolean'
                analysis.update({
                    'true_count': col_data.sum(),
                    'false_count': len(col_data) - col_data.sum()
                })

            type_analysis[column] = analysis

            # 출력
            print(f"📊 {column}:")
            print(f"   Type: {analysis['mcp_type']} ({analysis.get('sub_type', 'standard')})")
            print(f"   Nulls: {analysis['null_percentage']:.1f}%")
            print(f"   Unique: {analysis['unique_count']} ({analysis['unique_percentage']:.1f}%)")

        self.column_analysis = type_analysis
        return type_analysis

    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        2단계: 자동 전처리 파이프라인 생성
        데이터 타입에 따라 맞춤형 전처리 단계 생성
        """
        print("\n🔧 Step 2: Automatic Preprocessing Pipeline Creation")
        print("=" * 50)

        pipeline = []

        for column, analysis in self.column_analysis.items():
            mcp_type = analysis['mcp_type']

            # 결측치 처리
            if analysis['null_percentage'] > 0:
                if analysis['null_percentage'] > 50:
                    pipeline.append({
                        'step': 'remove_column',
                        'column': column,
                        'reason': f'High missing data: {analysis["null_percentage"]:.1f}%'
                    })
                    continue
                else:
                    # 타입별 결측치 처리 방법
                    if mcp_type == 'numeric':
                        method = 'median' if analysis.get('has_outliers') else 'mean'
                    elif mcp_type == 'categorical':
                        method = 'mode'
                    else:
                        method = 'forward_fill'

                    pipeline.append({
                        'step': 'handle_missing',
                        'column': column,
                        'method': method,
                        'reason': f'Fill {analysis["null_percentage"]:.1f}% missing values'
                    })

            # 타입별 전처리
            if mcp_type == 'numeric':
                # 이상치 처리
                if analysis.get('has_outliers'):
                    pipeline.append({
                        'step': 'handle_outliers',
                        'column': column,
                        'method': 'iqr_clipping',
                        'reason': 'Outliers detected'
                    })

                # 정규화/표준화
                if analysis['std'] > analysis['mean'] * 2:  # 높은 분산
                    pipeline.append({
                        'step': 'normalize',
                        'column': column,
                        'method': 'standard_scaling',
                        'reason': 'High variance detected'
                    })

            elif mcp_type == 'categorical':
                cardinality = analysis.get('cardinality_level')

                if cardinality == 'high':
                    pipeline.append({
                        'step': 'reduce_cardinality',
                        'column': column,
                        'method': 'top_categories',
                        'reason': f'High cardinality: {analysis["unique_count"]} categories'
                    })

                # 인코딩
                if analysis['unique_count'] <= 10:
                    encoding_method = 'one_hot'
                else:
                    encoding_method = 'label_encoding'

                pipeline.append({
                    'step': 'encode_categorical',
                    'column': column,
                    'method': encoding_method,
                    'reason': f'Categorical encoding for {analysis["unique_count"]} categories'
                })

        # 중복 제거
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            pipeline.append({
                'step': 'remove_duplicates',
                'reason': f'Found {duplicates} duplicate rows'
            })

        # 출력
        for step in pipeline:
            print(f"🔧 {step['step']}: {step.get('column', 'all')} - {step['reason']}")

        self.preprocessing_pipeline = pipeline
        return pipeline

    def create_analytics_pipeline(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        3단계: 자동 분석 파이프라인 생성
        데이터 특성에 따라 최적 분석 방법 선택
        """
        print("\n📊 Step 3: Automatic Analytics Pipeline Creation")
        print("=" * 50)

        analytics = []

        # 데이터 개요
        numeric_cols = [col for col, analysis in self.column_analysis.items()
                       if analysis['mcp_type'] == 'numeric']
        categorical_cols = [col for col, analysis in self.column_analysis.items()
                           if analysis['mcp_type'] == 'categorical']

        # 1. 기본 통계 분석
        analytics.append({
            'analysis': 'descriptive_statistics',
            'scope': 'all_columns',
            'reason': 'Basic data understanding'
        })

        # 2. 상관관계 분석 (숫자형 변수가 2개 이상)
        if len(numeric_cols) >= 2:
            analytics.append({
                'analysis': 'correlation_analysis',
                'scope': numeric_cols,
                'reason': f'Multiple numeric variables ({len(numeric_cols)})'
            })

        # 3. 분포 분석
        for col in numeric_cols:
            analysis = self.column_analysis[col]
            if analysis.get('has_outliers'):
                analytics.append({
                    'analysis': 'outlier_analysis',
                    'scope': [col],
                    'reason': f'Outliers detected in {col}'
                })

        # 4. 범주형 변수 분석
        for col in categorical_cols:
            analysis = self.column_analysis[col]
            if analysis['unique_count'] <= 20:  # 범주가 너무 많지 않으면
                analytics.append({
                    'analysis': 'frequency_analysis',
                    'scope': [col],
                    'reason': f'Categorical distribution for {col}'
                })

        # 5. 고급 분석 (데이터 크기와 특성에 따라)
        if len(df) > 100 and len(numeric_cols) >= 3:
            analytics.append({
                'analysis': 'clustering_analysis',
                'scope': numeric_cols,
                'method': 'kmeans',
                'reason': 'Sufficient data for pattern detection'
            })

        if len(numeric_cols) >= 4:
            analytics.append({
                'analysis': 'dimensionality_reduction',
                'scope': numeric_cols,
                'method': 'pca',
                'reason': 'Multiple dimensions for PCA'
            })

        # 6. 시각화
        for col in numeric_cols:
            analytics.append({
                'analysis': 'visualization',
                'type': 'histogram',
                'scope': [col],
                'reason': f'Distribution visualization for {col}'
            })

        if len(numeric_cols) >= 2:
            analytics.append({
                'analysis': 'visualization',
                'type': 'correlation_heatmap',
                'scope': numeric_cols,
                'reason': 'Correlation visualization'
            })

        # 출력
        for step in analytics:
            scope_str = step['scope'] if isinstance(step['scope'], str) else f"{len(step['scope'])} columns"
            print(f"📈 {step['analysis']}: {scope_str} - {step['reason']}")

        self.analytics_pipeline = analytics
        return analytics

    def _classify_numeric(self, series):
        """숫자형 세부 분류"""
        if series.dtype == 'int64' and series.nunique() <= 20:
            return 'ordinal'
        elif series.min() >= 0 and series.max() <= 1:
            return 'probability'
        elif series.dtype == 'int64':
            return 'integer'
        else:
            return 'continuous'

    def _classify_categorical(self, series):
        """범주형 세부 분류"""
        unique_count = series.nunique()
        if unique_count <= 5:
            return 'nominal_low'
        elif unique_count <= 20:
            return 'nominal_medium'
        else:
            return 'nominal_high'

    def _classify_cardinality(self, series):
        """카디널리티 분류"""
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.8:
            return 'high'
        elif unique_ratio > 0.3:
            return 'medium'
        else:
            return 'low'

    def _detect_outliers(self, series):
        """이상치 감지"""
        if series.dtype not in ['int64', 'float64']:
            return False
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).any()

    def _analyze_distribution(self, series):
        """분포 분석"""
        skewness = series.skew()
        if abs(skewness) < 0.5:
            return 'normal'
        elif skewness > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'

    def _might_be_datetime(self, series):
        """날짜/시간 가능성 체크"""
        # 간단한 날짜 패턴 체크
        sample = series.dropna().astype(str).head(10)
        date_like = 0
        for value in sample:
            if any(pattern in value for pattern in ['-', '/', ':', 'T']):
                date_like += 1
        return date_like > len(sample) * 0.5

def demo_automatic_processing():
    """실제 사용 예제"""
    print("🚀 MCP Automatic Data Processing Demo")
    print("=" * 60)

    # 샘플 데이터 생성
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': range(1, 1001),  # 정수형 ID
        'age': np.random.normal(35, 10, 1000).astype(int),  # 숫자형
        'income': np.random.lognormal(10, 0.5, 1000),  # 숫자형 (치우친 분포)
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),  # 범주형 (낮은 카디널리티)
        'region': np.random.choice([f'Region_{i}' for i in range(50)], 1000),  # 범주형 (높은 카디널리티)
        'is_premium': np.random.choice([True, False], 1000),  # 불린형
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D')  # 날짜형
    })

    # 일부 결측치 추가
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'category'] = np.nan

    print(f"📋 Sample Data Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print()

    # 자동 처리 시스템 실행
    processor = AutoDataProcessor()

    # 1. 자동 타입 판단
    type_analysis = processor.judge_data_types(df)

    # 2. 자동 전처리 파이프라인 생성
    preprocessing = processor.create_preprocessing_pipeline(df)

    # 3. 자동 분석 파이프라인 생성
    analytics = processor.create_analytics_pipeline(df)

    print("\n✅ MCP Automatic Processing Complete!")
    print(f"📊 Data Types Identified: {len(type_analysis)} columns")
    print(f"🔧 Preprocessing Steps: {len(preprocessing)} operations")
    print(f"📈 Analytics Steps: {len(analytics)} analyses")

    return {
        'data_types': type_analysis,
        'preprocessing_pipeline': preprocessing,
        'analytics_pipeline': analytics
    }

if __name__ == "__main__":
    result = demo_automatic_processing()

    # 결과를 JSON으로 저장 (실제 MCP에서 사용하는 형식)
    with open('automatic_pipeline_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Result saved to: automatic_pipeline_result.json")