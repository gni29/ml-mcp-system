#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

class MLMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'ml-mcp-high-performance',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  // 사용 가능한 도구 목록 정의
  async getAvailableTools() {
    return [
      {
        name: 'general_query',
        description: '일반적인 질문에 답변합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '질문 내용'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'system_status',
        description: '시스템 상태를 확인합니다.',
        inputSchema: {
          type: 'object',
          properties: {},
          required: []
        }
      },
      {
        name: 'analyze_data',
        description: '데이터 분석을 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '분석 요청 내용'
            },
            auto_detect_files: {
              type: 'boolean',
              description: '파일 자동 감지 여부'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'visualize_data',
        description: '데이터 시각화를 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '시각화 요청 내용'
            },
            auto_detect_files: {
              type: 'boolean',
              description: '파일 자동 감지 여부'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'train_model',
        description: '머신러닝 모델을 훈련합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '모델 훈련 요청 내용'
            },
            auto_detect_files: {
              type: 'boolean',
              description: '파일 자동 감지 여부'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'mode_switch',
        description: '작업 모드를 변경합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            mode: {
              type: 'string',
              enum: ['general', 'ml', 'data_analysis', 'visualization'],
              description: '변경할 모드'
            }
          },
          required: ['mode']
        }
      }
    ];
  }

  setupHandlers() {
    // 도구 목록 제공
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      try {
        const tools = await this.getAvailableTools();
        return { tools };
      } catch (error) {
        console.error('도구 목록 조회 실패:', error);
        return { tools: [] };
      }
    });

    // 도구 실행 처리
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        return await this.handleToolCall(name, args);
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `도구 실행 중 오류 발생: ${error.message}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async handleToolCall(toolName, args) {
    switch (toolName) {
      case 'general_query':
        return {
          content: [
            {
              type: 'text',
              text: `일반 질문 처리: "${args.query}"\n\n안녕하세요! 현재 기본 응답 모드로 실행 중입니다. 더 고급 기능을 사용하려면 Ollama 모델을 설정하고 필요한 컴포넌트를 구현해주세요.`
            }
          ]
        };

      case 'system_status':
        return {
          content: [
            {
              type: 'text',
              text: `🔍 시스템 상태 확인:

✅ MCP 서버: 실행 중
✅ 기본 도구: 사용 가능
⚠️ Ollama 모델: 구성 필요
⚠️ Python 스크립트: 구현 필요
⚠️ 데이터 분석: 제한적 지원

현재 시간: ${new Date().toLocaleString()}
프로세스 ID: ${process.pid}
메모리 사용량: ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`
            }
          ]
        };

      case 'analyze_data':
        return {
          content: [
            {
              type: 'text',
              text: `📊 데이터 분석 요청: "${args.query}"

현재 기본 모드로 실행 중입니다. 
실제 데이터 분석 기능을 사용하려면:
1. Python 환경 설정
2. 데이터 분석 스크립트 구현
3. 파일 처리 로직 추가

요청 내용을 기록했습니다.`
            }
          ]
        };

      case 'visualize_data':
        return {
          content: [
            {
              type: 'text',
              text: `📈 시각화 요청: "${args.query}"

현재 기본 모드로 실행 중입니다.
실제 시각화 기능을 사용하려면:
1. Python 시각화 라이브러리 설정
2. 차트 생성 스크립트 구현
3. 이미지 파일 생성 로직 추가

요청 내용을 기록했습니다.`
            }
          ]
        };

      case 'train_model':
        return {
          content: [
            {
              type: 'text',
              text: `🤖 모델 훈련 요청: "${args.query}"

현재 기본 모드로 실행 중입니다.
실제 모델 훈련 기능을 사용하려면:
1. 머신러닝 라이브러리 설정
2. 모델 훈련 스크립트 구현
3. 데이터 전처리 로직 추가

요청 내용을 기록했습니다.`
            }
          ]
        };

      case 'mode_switch':
        return {
          content: [
            {
              type: 'text',
              text: `🔄 모드 변경: ${args.mode}

현재는 기본 모드만 지원합니다.
요청하신 '${args.mode}' 모드는 추후 구현 예정입니다.

모드 변경 요청을 기록했습니다.`
            }
          ]
        };

      default:
        return {
          content: [
            {
              type: 'text',
              text: `알 수 없는 도구: ${toolName}`
            }
          ],
          isError: true
        };
    }
  }

  async run() {
    try {
      console.error('MCP 서버 초기화 중...');
      
      // MCP 서버 시작
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      
      // 서버 시작 메시지 (CLI에서 감지할 수 있도록 stdout 사용)
      console.log('ML MCP 서버가 시작되었습니다.');
      
    } catch (error) {
      console.error('서버 시작 실패:', error.message);
      process.exit(1);
    }
  }
}

// 서버 시작
const server = new MLMCPServer();
server.run().catch(console.error);
