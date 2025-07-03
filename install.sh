#!/bin/bash

echo "🚀 고성능 ML MCP 시스템 설치 시작..."

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 에러 처리
set -e

# 함수: 에러 메시지 출력
error_exit() {
    echo -e "${RED}❌ 오류: $1${NC}" >&2
    exit 1
}

# 함수: 성공 메시지 출력
success_msg() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 함수: 경고 메시지 출력
warning_msg() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 함수: 정보 메시지 출력
info_msg() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 1. 시스템 요구사항 확인
echo ""
echo "📋 시스템 요구사항 확인 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Node.js 확인
if ! command -v node &> /dev/null; then
    error_exit "Node.js가 설치되지 않았습니다. https://nodejs.org에서 설치하세요."
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    error_exit "Node.js 18 이상이 필요합니다. 현재 버전: $(node --version)"
fi

success_msg "Node.js 버전 확인: $(node --version)"

# npm 확인
if ! command -v npm &> /dev/null; then
    error_exit "npm이 설치되지 않았습니다."
fi

success_msg "npm 버전 확인: $(npm --version)"

# Python 확인
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3가 설치되지 않았습니다."
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
success_msg "Python 버전 확인: $(python3 --version)"

# pip 확인
if ! command -v pip3 &> /dev/null; then
    warning_msg "pip3가 설치되지 않았습니다. Python 패키지 설치에 문제가 있을 수 있습니다."
else
    success_msg "pip3 버전 확인: $(pip3 --version | cut -d' ' -f2)"
fi

# curl 확인
if ! command -v curl &> /dev/null; then
    error_exit "curl이 설치되지 않았습니다."
fi

# 메모리 확인 (Linux/macOS)
if command -v free &> /dev/null; then
    # Linux
    TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
    if [ "$TOTAL_RAM" -lt 8 ]; then
        warning_msg "권장 RAM은 8GB 이상입니다. 현재: ${TOTAL_RAM}GB"
        echo "최소 8GB (권장 32GB)의 RAM이 필요합니다."
        read -p "계속하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        success_msg "메모리 확인: ${TOTAL_RAM}GB"
    fi
elif command -v sysctl &> /dev/null; then
    # macOS
    TOTAL_RAM_BYTES=$(sysctl -n hw.memsize)
    TOTAL_RAM_GB=$((TOTAL_RAM_BYTES / 1024 / 1024 / 1024))
    if [ "$TOTAL_RAM_GB" -lt 8 ]; then
        warning_msg "권장 RAM은 8GB 이상입니다. 현재: ${TOTAL_RAM_GB}GB"
        read -p "계속하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        success_msg "메모리 확인: ${TOTAL_RAM_GB}GB"
    fi
else
    warning_msg "메모리 확인을 건너뜁니다."
fi

# 디스크 공간 확인
AVAILABLE_SPACE=$(df -h . | awk 'NR==2{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 15 ]; then
    warning_msg "권장 디스크 공간은 15GB 이상입니다. 현재: ${AVAILABLE_SPACE}GB"
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    success_msg "디스크 공간 확인: ${AVAILABLE_SPACE}GB 사용 가능"
fi

success_msg "시스템 요구사항 확인 완료"

# 2. Ollama 설치 확인
echo ""
echo "🦙 Ollama 설치 확인 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v ollama &> /dev/null; then
    warning_msg "Ollama가 설치되지 않았습니다."
    echo "Ollama를 설치하시겠습니까? (권장)"
    read -p "설치하기 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info_msg "Ollama 설치 중..."
        
        # OS별 설치 방법
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -fsSL https://ollama.com/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install ollama
            else
                curl -fsSL https://ollama.com/install.sh | sh
            fi
        else
            warning_msg "자동 설치를 지원하지 않는 OS입니다."
            echo "https://ollama.com에서 수동으로 설치하세요."
            exit 1
        fi
        
        success_msg "Ollama 설치 완료"
    else
        warning_msg "Ollama를 수동으로 설치해야 합니다: https://ollama.com"
    fi
else
    success_msg "Ollama 설치 확인: $(ollama --version)"
fi

# 3. 프로젝트 설정
echo ""
echo "📦 프로젝트 설정 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 현재 디렉토리 확인
if [[ ! -f "package.json" ]]; then
    error_exit "package.json 파일이 없습니다. 프로젝트 디렉토리에서 실행하세요."
fi

info_msg "현재 디렉토리: $(pwd)"

# 필요한 디렉토리 생성
info_msg "필요한 디렉토리 생성 중..."
mkdir -p results
mkdir -p uploads
mkdir -p temp
mkdir -p logs
mkdir -p data/{state,cache,logs}
mkdir -p models
mkdir -p scripts
mkdir -p test
mkdir -p test_results
mkdir -p config

success_msg "디렉토리 생성 완료"

# 4. Node.js 의존성 설치
echo ""
echo "📚 Node.js 의존성 설치 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

info_msg "npm 의존성 설치 중..."
npm install --prefer-offline --no-audit --progress=false

success_msg "Node.js 의존성 설치 완료"

# 5. Python 가상환경 설정
echo ""
echo "🐍 Python 환경 설정 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Python 가상환경 생성 (선택사항)
if [ ! -d "python-env" ]; then
    echo "Python 가상환경을 생성하시겠습니까? (권장)"
    read -p "생성하기 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info_msg "Python 가상환경 생성 중..."
        python3 -m venv python-env
        
        # 가상환경 활성화
        source python-env/bin/activate
        
        # pip 업그레이드
        pip install --upgrade pip
        
        success_msg "Python 가상환경 생성 완료"
    else
        info_msg "시스템 Python 사용"
    fi
else
    success_msg "Python 가상환경 이미 존재"
fi

# 6. Python 패키지 설치
info_msg "Python 패키지 설치 중..."

# 필수 패키지 목록
PYTHON_PACKAGES=(
    "pandas>=1.3.0"
    "numpy>=1.21.0"
    "scikit-learn>=1.0.0"
    "matplotlib>=3.5.0"
    "seaborn>=0.11.0"
    "plotly>=5.0.0"
    "openpyxl>=3.0.0"
    "xlrd>=2.0.0"
    "pyarrow>=5.0.0"
    "h5py>=3.0.0"
    "jupyter>=1.0.0"
)

# 패키지 설치
for package in "${PYTHON_PACKAGES[@]}"; do
    info_msg "설치 중: $package"
    if [ -d "python-env" ]; then
        python-env/bin/pip install "$package" --quiet
    else
        pip3 install "$package" --quiet --user
    fi
done

success_msg "Python 패키지 설치 완료"

# 7. 실행 권한 설정
echo ""
echo "🔐 실행 권한 설정 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

chmod +x mcp-cli.js
chmod +x main.js
chmod +x install.sh
chmod +x setup-mcp-cli.sh

# scripts 디렉토리의 스크립트들
if [ -f "scripts/setup-models.js" ]; then
    chmod +x scripts/setup-models.js
fi

success_msg "실행 권한 설정 완료"

# 8. 설정 파일 생성
echo ""
echo "⚙️ 설정 파일 생성 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 환경 변수 템플릿 생성
if [ ! -f ".env.example" ]; then
    cat > .env.example << 'EOF'
# ML MCP 시스템 환경 변수
NODE_ENV=production
LOG_LEVEL=info
OLLAMA_ENDPOINT=http://localhost:11434
ROUTER_MODEL=llama3.2:3b
PROCESSOR_MODEL=qwen2.5:14b
MAX_MEMORY_MB=32000
CACHE_SIZE_MB=1000
AUTO_UNLOAD_TIMEOUT=600000
EOF
    success_msg ".env.example 파일 생성 완료"
fi

# 기본 .env 파일 생성 (존재하지 않는 경우)
if [ ! -f ".env" ]; then
    cp .env.example .env
    success_msg ".env 파일 생성 완료"
fi

# 메모리 임계값 설정 파일 생성
if [ ! -f "config/memory-thresholds.json" ]; then
    cat > config/memory-thresholds.json << 'EOF'
{
  "router": {
    "maxMemoryMB": 6000,
    "warningThresholdMB": 5000,
    "autoUnload": false,
    "autoUnloadTimeoutMs": 300000
  },
  "processor": {
    "maxMemoryMB": 28000,
    "warningThresholdMB": 25000,
    "autoUnload": true,
    "autoUnloadTimeoutMs": 600000
  },
  "system": {
    "maxTotalMemoryMB": 32000,
    "emergencyThresholdMB": 30000,
    "warningThresholdMB": 25000,
    "criticalThresholdMB": 28000
  },
  "cache": {
    "maxSizeMB": 1000,
    "warningThresholdMB": 800,
    "cleanupPercentage": 0.3
  },
  "cleanup": {
    "minIntervalMs": 60000,
    "maxTempFileAge": 3600000,
    "maxLogFileAge": 86400000
  }
}
EOF
    success_msg "메모리 임계값 설정 파일 생성 완료"
fi

# 9. Ollama 서비스 상태 확인
echo ""
echo "🔍 Ollama 서비스 상태 확인 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ollama 서비스 시작 시도
if command -v ollama &> /dev/null; then
    # 서비스 실행 중인지 확인
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        warning_msg "Ollama 서비스가 실행되지 않고 있습니다."
        echo "Ollama 서비스를 시작하시겠습니까?"
        read -p "시작하기 (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info_msg "Ollama 서비스 시작 중..."
            
            # 백그라운드에서 Ollama 서비스 시작
            nohup ollama serve > /dev/null 2>&1 &
            
            # 서비스 시작 대기
            sleep 5
            
            # 재확인
            if curl -s http://localhost:11434/api/version &> /dev/null; then
                success_msg "Ollama 서비스 시작 완료"
            else
                warning_msg "Ollama 서비스 시작에 실패했습니다."
                echo "수동으로 'ollama serve' 명령을 실행하세요."
            fi
        else
            warning_msg "Ollama 서비스를 수동으로 시작해야 합니다: ollama serve"
        fi
    else
        success_msg "Ollama 서비스가 실행 중입니다."
    fi
else
    warning_msg "Ollama가 설치되지 않았습니다."
fi

# 10. 모델 설치 확인
echo ""
echo "🤖 필요한 모델 확인 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if curl -s http://localhost:11434/api/version &> /dev/null; then
    # 설치된 모델 확인
    info_msg "설치된 모델 확인 중..."
    
    MODELS_INSTALLED=true
    
    if curl -s http://localhost:11434/api/tags | grep -q "llama3.2:3b"; then
        success_msg "Llama 3.2 3B 모델 확인됨"
    else
        warning_msg "Llama 3.2 3B 모델이 설치되지 않았습니다."
        MODELS_INSTALLED=false
    fi
    
    if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:14b"; then
        success_msg "Qwen 2.5 14B 모델 확인됨"
    else
        warning_msg "Qwen 2.5 14B 모델이 설치되지 않았습니다."
        MODELS_INSTALLED=false
    fi
    
    if [ "$MODELS_INSTALLED" = false ]; then
        echo ""
        echo "필요한 모델을 자동으로 설치하시겠습니까?"
        echo "⚠️  주의: 모델 다운로드는 시간이 오래 걸릴 수 있습니다 (~10GB)"
        read -p "설치하기 (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info_msg "모델 설치 시작..."
            if [ -f "scripts/setup-models.js" ]; then
                node scripts/setup-models.js
            else
                warning_msg "모델 설치 스크립트가 없습니다. 수동으로 설치하세요:"
                echo "  ollama pull llama3.2:3b"
                echo "  ollama pull qwen2.5:14b"
            fi
        else
            warning_msg "나중에 다음 명령으로 모델을 설치하세요:"
            echo "  npm run models"
            echo "  또는 수동으로:"
            echo "  ollama pull llama3.2:3b"
            echo "  ollama pull qwen2.5:14b"
        fi
    fi
else
    warning_msg "Ollama 서비스가 실행되지 않고 있어 모델 확인을 건너뜁니다."
fi

# 11. 설치 테스트
echo ""
echo "🧪 설치 테스트 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 기본 모듈 import 테스트
info_msg "Node.js 모듈 테스트 중..."
if timeout 10s node -e "import './main.js'; console.log('✅ 모듈 로드 성공');" 2>/dev/null; then
    success_msg "Node.js 모듈 테스트 성공"
else
    warning_msg "Node.js 모듈 테스트 실패 - 일부 기능이 제한될 수 있습니다."
fi

# Python 환경 테스트
info_msg "Python 환경 테스트 중..."
if [ -d "python-env" ]; then
    if python-env/bin/python -c "import pandas, numpy, sklearn; print('✅ Python 패키지 로드 성공')" 2>/dev/null; then
        success_msg "Python 환경 테스트 성공"
    else
        warning_msg "Python 환경 테스트 실패"
    fi
else
    if python3 -c "import pandas, numpy, sklearn; print('✅ Python 패키지 로드 성공')" 2>/dev/null; then
        success_msg "Python 환경 테스트 성공"
    else
        warning_msg "Python 환경 테스트 실패"
    fi
fi

# 12. 설치 완료 및 다음 단계 안내
echo ""
echo "🎉 설치 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

success_msg "ML MCP 시스템 설치가 완료되었습니다!"

echo ""
echo "🚀 다음 단계:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 📋 Ollama 서비스 실행 (필요시):"
echo "     ollama serve"
echo ""
echo "2. 🤖 모델 설치 (필요시):"
echo "     npm run models"
echo "     또는 수동으로:"
echo "     ollama pull llama3.2:3b"
echo "     ollama pull qwen2.5:14b"
echo ""
echo "3. 🧪 시스템 테스트:"
echo "     npm run test"
echo ""
echo "4. 🔧 CLI 실행:"
echo "     npm run cli"
echo ""
echo "5. 📊 MCP 서버 실행:"
echo "     npm run start"
echo ""

echo "💡 유용한 명령어:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• npm run cli      - 대화형 CLI 실행"
echo "• npm run start    - MCP 서버 실행"
echo "• npm run models   - 모델 자동 설치"
echo "• npm run test     - 통합 테스트"
echo "• npm run setup    - 이 설치 스크립트 재실행"
echo ""

echo "📚 추가 정보:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• 설정 파일: .env, config/memory-thresholds.json"
echo "• 로그 파일: logs/ 디렉토리"
echo "• 결과 파일: results/ 디렉토리"
echo "• 문제 해결: README.md 참조"
echo ""

echo "🎯 시작하기:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "가장 빠른 시작 방법:"
echo "1. ollama serve           (새 터미널)"
echo "2. npm run models         (모델 설치)"
echo "3. npm run cli            (CLI 실행)"
echo ""

success_msg "설치 스크립트 완료!"
echo "문제가 발생하면 GitHub Issues 또는 문서를 참고하세요."
