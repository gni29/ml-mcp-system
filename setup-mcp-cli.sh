#!/bin/bash

echo "🚀 MCP CLI 설정 시작..."

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 에러 처리
set -e

error_exit() {
    echo -e "${RED}❌ 오류: $1${NC}" >&2
    exit 1
}

success_msg() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning_msg() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info_msg() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 1. 현재 디렉토리 확인
if [[ ! -f "package.json" ]]; then
    error_exit "package.json 파일이 없습니다. ml-mcp-system 디렉토리에서 실행하세요."
fi

info_msg "현재 디렉토리: $(pwd)"

# 2. Node.js 및 npm 확인
if ! command -v node &> /dev/null; then
    error_exit "Node.js가 설치되지 않았습니다. https://nodejs.org에서 설치하세요."
fi

if ! command -v npm &> /dev/null; then
    error_exit "npm이 설치되지 않았습니다."
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    error_exit "Node.js 18 이상이 필요합니다. 현재 버전: $(node --version)"
fi

success_msg "Node.js 버전: $(node --version)"

# 3. 필요한 의존성 설치
info_msg "npm 의존성 설치 중..."
npm install

success_msg "의존성 설치 완료"

# 4. 필요한 디렉토리 생성
info_msg "필요한 디렉토리 생성 중..."
mkdir -p results
mkdir -p uploads
mkdir -p temp
mkdir -p logs
mkdir -p data/{state,cache,logs}

success_msg "디렉토리 생성 완료"

# 5. 실행 권한 부여
info_msg "실행 권한 설정 중..."
chmod +x mcp-cli.js
chmod +x main.js

success_msg "실행 권한 설정 완료"

# 6. Ollama 서비스 확인
info_msg "Ollama 서비스 확인 중..."
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    warning_msg "Ollama 서비스가 실행되지 않고 있습니다."
    echo "다음 명령어로 Ollama를 시작하세요:"
    echo "  ollama serve"
    echo ""
    echo "그 다음 필요한 모델을 다운로드하세요:"
    echo "  ollama pull llama3.2:3b"
    echo "  ollama pull qwen2.5:14b"
else
    success_msg "Ollama 서비스가 실행 중입니다."
    
    # 모델 확인
    info_msg "설치된 모델 확인 중..."
    if curl -s http://localhost:11434/api/tags | grep -q "llama3.2:3b"; then
        success_msg "Llama 3.2 3B 모델 확인됨"
    else
        warning_msg "Llama 3.2 3B 모델이 설치되지 않았습니다."
        echo "다음 명령어로 설치하세요: ollama pull llama3.2:3b"
    fi
    
    if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:14b"; then
        success_msg "Qwen 2.5 14B 모델 확인됨"
    else
        warning_msg "Qwen 2.5 14B 모델이 설치되지 않았습니다."
        echo "다음 명령어로 설치하세요: ollama pull qwen2.5:14b"
    fi
fi

# 7. 테스트 실행
info_msg "연결 테스트 중..."
timeout 10s node -e "
import { ModelManager } from './core/model-manager.js';
const manager = new ModelManager();
manager.initialize().then(() => {
    console.log('✅ MCP 서버 연결 테스트 성공');
    process.exit(0);
}).catch(err => {
    console.error('❌ MCP 서버 연결 테스트 실패:', err.message);
    process.exit(1);
});
" 2>/dev/null || warning_msg "연결 테스트 실패 - Ollama 서비스와 모델을 확인하세요."

# 8. 완료 메시지
echo ""
echo "🎉 MCP CLI 설정이 완료되었습니다!"
echo ""
echo "📋 사용법:"
echo "  1. MCP CLI 실행: npm run cli"
echo "  2. 또는 직접 실행: node mcp-cli.js"
echo "  3. 또는 전역 명령: ./mcp-cli.js"
echo ""
echo "🔧 필요한 사전 작업:"
echo "  - Ollama 서비스 실행: ollama serve"
echo "  - 모델 다운로드: ollama pull llama3.2:3b && ollama pull qwen2.5:14b"
echo ""
echo "📖 사용 예시:"
echo "  ML> 안녕하세요"
echo "  ML> data.csv 파일을 분석해주세요"
echo "  ML> 이 데이터로 예측 모델을 만들어주세요"
echo "  ML> 시각화 차트를 그려주세요"
echo ""
echo "🚪 종료: 'exit' 또는 'Ctrl+C'"
echo ""
success_msg "설정 완료! npm run cli 명령으로 시작하세요."
