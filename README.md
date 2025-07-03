# README.md 파일 업데이트
cat > README.md << 'EOF'
# ML MCP System

고성능 머신러닝/데이터분석 MCP(Model Context Protocol) 시스템

## 🎯 특징

- **이중 모델 시스템**: Llama 3.2 3B (라우팅) + Qwen2.5 14B (메인 작업)
- **장비 독립적**: 어떤 하드웨어에서든 동일한 성능
- **토글 모드**: 일반 모드 ↔ ML 모드 자유 전환
- **완전 로컬**: 모든 처리가 로컬에서 실행

## 🚀 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/your-username/ml-mcp-system.git
cd ml-mcp-system

# 자동 설치
chmod +x install.sh
./install.sh

# Claude Desktop 재시작 후 사용
