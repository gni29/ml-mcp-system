/**
 * ML MCP Shared Utilities
 * Export all shared components
 */

export { Logger } from './utils/logger.js';
export { BaseService } from './utils/base-service.js';

// Common constants
export const MCP_VERSIONS = {
  ANALYSIS: '1.0.0',
  ML: '1.0.0',
  VISUALIZATION: '1.0.0',
  SHARED: '1.0.0'
};

export const COMMON_SCHEMAS = {
  DATA_FILE: {
    type: 'string',
    description: '분석할 데이터 파일 경로'
  },
  OUTPUT_DIR: {
    type: 'string',
    description: '출력 디렉토리',
    default: 'results'
  }
};