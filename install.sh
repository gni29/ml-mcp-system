#!/bin/bash

echo "🚀 고성능 ML MCP 시스템 설치 시작..."

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 1. 시스템 요구사항 확인
echo "📋 시스템 요구사항 확인 중..."

# Node.js 확인
if ! command -v node &> /dev/null; then
    error_exit "Node.js가 설치되지 않았습니다. https://nodejs.org에서 설치하세요."
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    error_exit "Node.js 18 이상이 필요합니다. 현재 버전: $(node --version)"
fi

success_msg "Node.js 버전 확인: $(node --version)"

# Python 확인
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3가 설치되지 않았습니다."
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
success_msg "Python 버전 확인: $(python3 --version)"

# 메모리 확인
TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
if [ "$TOTAL_RAM" -lt 32 ]; then
    warning_msg "권장 RAM은 32GB 이상입니다. 현재: ${TOTAL_RAM}GB"
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

success_msg "시스템 요구사항 확인 완료"

## 🚀 빠른 시작 가이드

### 1. 설치 실행

```bash
# 저장소 클론 (또는 파일들 수동 생성)
git clone <your-repo>/ml-mcp-system
cd ml-mcp-system

# 실행 권한 부여
chmod +x install.sh

# 자동 설치 실행
./install.sh
