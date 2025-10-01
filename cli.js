#!/usr/bin/env node

/**
 * ML MCP CLI - Standalone Command Line Interface
 * 머신러닝 데이터 분석을 위한 CLI 도구
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MLAnalysisCLI {
    constructor() {
        this.pythonRunner = path.join(__dirname, 'scripts', 'python_runner.py');
        this.dataDir = path.join(__dirname, 'data');
        this.outputDir = path.join(__dirname, 'results');
    }

    async ensureDirectories() {
        try {
            await fs.mkdir(this.dataDir, { recursive: true });
            await fs.mkdir(this.outputDir, { recursive: true });
        } catch (error) {
            // 디렉토리가 이미 존재하는 경우 무시
        }
    }

    async runPythonScript(command, args = []) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [this.pythonRunner, command, ...args], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: __dirname
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(stdout.trim());
                } else {
                    reject(new Error(`Python 스크립트 실행 실패: ${stderr}`));
                }
            });
        });
    }

    async listModules() {
        console.log('🔍 사용 가능한 분석 모듈 목록:');
        console.log('=' .repeat(50));

        try {
            const result = await this.runPythonScript('list');
            console.log(result);
        } catch (error) {
            console.error('❌ 모듈 목록 조회 실패:', error.message);
        }
    }

    async validateSystem() {
        console.log('🔧 시스템 유효성 검사 중...');
        console.log('=' .repeat(50));

        try {
            const result = await this.runPythonScript('validate');
            console.log(result);
        } catch (error) {
            console.error('❌ 시스템 검증 실패:', error.message);
        }
    }

    async analyzeData(filePath, analysisType = 'basic', outputDir = null) {
        if (!filePath) {
            console.error('❌ 데이터 파일 경로가 필요합니다.');
            return;
        }

        const absPath = path.resolve(filePath);
        const outputPath = outputDir || this.outputDir;

        console.log(`📊 데이터 분석 시작: ${path.basename(filePath)}`);
        console.log(`📁 출력 디렉토리: ${outputPath}`);
        console.log('🔄 HTML 리포트와 JSON 결과를 생성합니다...');
        console.log('=' .repeat(50));

        try {
            await this.ensureDirectories();

            // 통합 분석기를 사용하여 HTML 리포트와 JSON 결과를 동시에 생성
            const args = [
                '--data', absPath,
                '--output', outputPath,
                '--format', 'both'  // HTML과 JSON 모두 생성
            ];

            if (analysisType !== 'basic') {
                args.push('--type', analysisType);
            }

            // integrated_analyzer.py를 사용하여 완전한 분석 수행
            const result = await this.runPythonScript('integrated', args);
            console.log(result);

            // 생성된 파일들 안내
            console.log('\n📋 생성된 파일들:');
            console.log(`  • HTML 리포트: ${outputPath}/analysis_report_*.html`);
            console.log(`  • JSON 결과: ${outputPath}/analysis_results_*.json`);
            console.log(`  • 시각화 이미지: ${outputPath}/plots/`);
            console.log('\n✅ 분석 완료!');
        } catch (error) {
            console.error('❌ 데이터 분석 실패:', error.message);
        }
    }

    async batchAnalysis(dataDir, outputDir = null) {
        if (!dataDir) {
            console.error('❌ 데이터 디렉토리 경로가 필요합니다.');
            return;
        }

        const outputPath = outputDir || path.join(this.outputDir, 'batch_results');

        console.log(`📦 배치 분석 시작: ${dataDir}`);
        console.log(`📁 출력 디렉토리: ${outputPath}`);
        console.log('=' .repeat(50));

        try {
            await this.ensureDirectories();

            const result = await this.runPythonScript('batch', [
                '--data', path.resolve(dataDir),
                '--output', outputPath
            ]);

            console.log(result);
            console.log('\n✅ 배치 분석 완료!');
        } catch (error) {
            console.error('❌ 배치 분석 실패:', error.message);
        }
    }

    showHelp() {
        console.log(`
🚀 ML MCP CLI - 머신러닝 데이터 분석 도구
=${'='.repeat(48)}

사용법:
  node cli.js <command> [options]

명령어:
  list                     사용 가능한 모든 분석 모듈 목록 표시
  validate                 시스템 및 모듈 유효성 검사
  help                     이 도움말 표시

데이터 분석:
  analyze <file>           데이터 분석 수행 (HTML 리포트 + JSON 결과 + 시각화)
    --type <type>          분석 유형 (clustering, pca, outlier_detection)
    --output <dir>         출력 디렉토리 (기본값: ./results)

  batch <directory>        디렉토리 내 모든 데이터 파일 배치 분석
    --output <dir>         출력 디렉토리 (기본값: ./results/batch_results)

예제:
  node cli.js list                                    # 모듈 목록 보기
  node cli.js validate                                # 시스템 검증
  node cli.js analyze data/sales.csv                  # 기본 분석
  node cli.js analyze data/customers.csv --type clustering  # 클러스터링 분석
  node cli.js batch data/                             # 배치 분석

지원하는 데이터 형식:
  • CSV (.csv)      • JSON (.json)     • Excel (.xlsx, .xls)
  • Parquet (.parquet)  • HDF5 (.h5, .hdf5)  • TSV (.tsv)

더 많은 정보: README.md 파일을 참조하세요.
`);
    }

    async run() {
        const args = process.argv.slice(2);

        if (args.length === 0) {
            this.showHelp();
            return;
        }

        const command = args[0];

        switch (command) {
            case 'list':
                await this.listModules();
                break;

            case 'validate':
                await this.validateSystem();
                break;

            case 'analyze':
                if (args.length < 2) {
                    console.error('❌ 분석할 파일 경로를 지정해주세요.');
                    console.log('사용법: node cli.js analyze <file> [--type <type>] [--output <dir>]');
                    return;
                }

                const filePath = args[1];
                const typeIndex = args.indexOf('--type');
                const outputIndex = args.indexOf('--output');

                const analysisType = typeIndex !== -1 && args[typeIndex + 1] ? args[typeIndex + 1] : 'basic';
                const outputDir = outputIndex !== -1 && args[outputIndex + 1] ? args[outputIndex + 1] : null;

                await this.analyzeData(filePath, analysisType, outputDir);
                break;

            case 'batch':
                if (args.length < 2) {
                    console.error('❌ 배치 분석할 디렉토리 경로를 지정해주세요.');
                    console.log('사용법: node cli.js batch <directory> [--output <dir>]');
                    return;
                }

                const dataDir = args[1];
                const batchOutputIndex = args.indexOf('--output');
                const batchOutputDir = batchOutputIndex !== -1 && args[batchOutputIndex + 1] ? args[batchOutputIndex + 1] : null;

                await this.batchAnalysis(dataDir, batchOutputDir);
                break;

            case 'help':
            case '--help':
            case '-h':
                this.showHelp();
                break;

            default:
                console.error(`❌ 알 수 없는 명령어: ${command}`);
                console.log('사용 가능한 명령어를 보려면 "node cli.js help"를 실행하세요.');
        }
    }
}

// CLI 실행
const cli = new MLAnalysisCLI();
cli.run().catch(error => {
    console.error('❌ CLI 실행 중 오류:', error.message);
    process.exit(1);
});

export default MLAnalysisCLI;