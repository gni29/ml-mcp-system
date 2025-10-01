# ML Visualization MCP Server
머신러닝 데이터 시각화 MCP 서버

## 개요 (Overview)

ML Visualization MCP Server는 머신러닝 및 데이터 분석을 위한 포괄적인 시각화 도구를 제공하는 MCP 서버입니다. 다양한 차트와 플롯을 생성하여 데이터 인사이트를 시각적으로 표현할 수 있습니다.

## 주요 기능 (Features)

### 📊 기본 차트 (Basic Charts)
- **선 그래프 (Line Chart)**: 시계열 데이터 및 연속 데이터 시각화
- **산점도 (Scatter Plot)**: 변수 간 상관관계 분석
- **히스토그램 (Histogram)**: 데이터 분포 분석
- **막대 그래프 (Bar Chart)**: 범주형 데이터 비교

### 📈 통계 플롯 (Statistical Plots)
- **박스 플롯 (Box Plot)**: 분포 및 이상값 탐지
- **상관관계 히트맵 (Correlation Heatmap)**: 변수 간 상관관계 매트릭스

### 🤖 머신러닝 시각화 (ML Visualizations)
- **혼동 행렬 (Confusion Matrix)**: 분류 모델 성능 평가
- **특성 중요도 (Feature Importance)**: 모델 특성 중요도 분석
- **학습 곡선 (Learning Curve)**: 모델 훈련 과정 분석

### 🎨 자동 시각화 (Auto Visualization)
- **자동 플롯 생성**: 데이터 타입에 따른 최적 시각화 자동 선택

### 🌐 대화형 차트 (Interactive Charts)
- **Plotly 기반**: 상호작용 가능한 웹 차트 생성

## 설치 및 실행 (Installation & Usage)

### 1. 의존성 설치
```bash
cd visualization-mcp
npm install
```

### 2. Python 의존성
```bash
pip install matplotlib seaborn pandas numpy scikit-learn plotly
```

### 3. 서버 실행
```bash
npm start
```

## 도구 목록 (Available Tools)

### 기본 차트 도구
- `create_line_chart`: 선 그래프 생성
- `create_scatter_plot`: 산점도 생성
- `create_histogram`: 히스토그램 생성
- `create_bar_chart`: 막대 그래프 생성

### 통계 플롯 도구
- `create_box_plot`: 박스 플롯 생성
- `create_correlation_heatmap`: 상관관계 히트맵 생성

### ML 시각화 도구
- `create_confusion_matrix`: 혼동 행렬 생성
- `create_feature_importance_plot`: 특성 중요도 플롯 생성
- `create_learning_curve`: 학습 곡선 생성

### 고급 도구
- `auto_visualize`: 자동 시각화
- `create_interactive_chart`: 대화형 차트 생성

## 사용 예시 (Examples)

### 히스토그램 생성
```json
{
  "data": {
    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  },
  "column": "values",
  "title": "데이터 분포",
  "bins": 20
}
```

### 혼동 행렬 생성
```json
{
  "y_true": [0, 1, 2, 0, 1, 2],
  "y_pred": [0, 1, 1, 0, 2, 2],
  "labels": ["클래스 A", "클래스 B", "클래스 C"]
}
```

### 특성 중요도 플롯
```json
{
  "feature_names": ["특성1", "특성2", "특성3"],
  "importance_values": [0.5, 0.3, 0.2],
  "top_k": 10
}
```

## 출력 형식 (Output Format)

모든 도구는 다음 형식의 JSON 응답을 반환합니다:

```json
{
  "success": true,
  "output_path": "chart.png",
  "chart_type": "histogram",
  "statistics": { ... },
  "insights": ["인사이트 1", "인사이트 2"],
  "analysis": { ... }
}
```

## 특징 (Features)

### 🇰🇷 한국어 지원
- 모든 라벨, 제목, 인사이트가 한국어로 제공
- 한국어 폰트 자동 설정

### 📊 상세한 분석
- 각 차트마다 통계 정보 제공
- 데이터 인사이트 자동 생성
- 이상값 탐지 및 패턴 분석

### 🎨 커스터마이징
- 색상 팔레트 선택
- 차트 크기 조정
- 다양한 스타일 옵션

### 🔧 강력한 에러 처리
- 상세한 에러 메시지
- 데이터 검증
- 대체 옵션 제공

## 기술 스택 (Tech Stack)

- **Backend**: Node.js, MCP SDK
- **Visualization**: Python, Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy
- **ML Metrics**: Scikit-learn

## 파일 구조 (File Structure)

```
visualization-mcp/
├── src/
│   └── index.js              # MCP 서버 메인 파일
├── package.json              # Node.js 패키지 정보
└── README.md                 # 이 파일

python/visualization/
├── 2d/
│   ├── histogram.py          # 히스토그램 생성기
│   ├── bar_chart.py          # 막대 그래프 생성기
│   ├── line.py               # 선 그래프 생성기
│   └── scatter_enhanced.py   # 향상된 산점도
├── statistical/
│   └── box_plot.py           # 박스 플롯 생성기
├── ml/
│   ├── confusion_matrix.py   # 혼동 행렬
│   ├── feature_importance.py # 특성 중요도
│   └── learning_curve.py     # 학습 곡선
├── interactive/
│   └── plotly_charts.py      # 대화형 차트
└── auto_visualizer.py        # 자동 시각화
```

## 성능 최적화 (Performance)

- 대용량 데이터셋 처리 최적화
- 메모리 효율적인 차트 생성
- 고해상도 이미지 출력 (300 DPI)
- 빠른 JSON 직렬화

## 확장성 (Extensibility)

새로운 시각화 타입을 쉽게 추가할 수 있도록 설계되었습니다:

1. Python 스크립트 작성
2. MCP 서버에 도구 등록
3. JSON 스키마 정의

## 라이센스 (License)

MIT License - 자세한 내용은 프로젝트 루트의 LICENSE 파일을 참조하세요.

## 기여하기 (Contributing)

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 지원 (Support)

문의사항이나 도움이 필요하시면 프로젝트 이슈를 생성해 주세요.